# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import importlib.util
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument(
    "--video", action="store_true", default=False, help="Record videos during training."
)
parser.add_argument(
    "--video_length",
    type=int,
    default=200,
    help="Length of the recorded video (in steps).",
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument(
    "--num_envs", type=int, default=None, help="Number of environments to simulate."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--center", action="store_true", default=False, help="Look at the robot."
)
parser.add_argument(
    "--pretrained",
    action="store_true",
    default=False,
    help="Load the exported TorchScript policy from assets instead of a training checkpoint.",
)
parser.add_argument(
    "--pretrained_path",
    type=str,
    default=None,
    help="Path to a TorchScript policy (.pt).",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# fail fast if rsl_rl is missing to avoid launching Isaac Sim
if importlib.util.find_spec("rsl_rl") is None:
    print(
        "[ERROR] Missing Python package `rsl_rl`.\n"
        "Install it in the same Python environment as Isaac Lab, e.g.:\n"
        "  pip install rsl-rl-lib"
    )
    raise SystemExit(1)

if importlib.util.find_spec("cat_envs") is None:
    repo_root = Path(__file__).resolve().parents[2]
    local_ext = repo_root / "exts" / "cat_envs"
    if local_ext.exists():
        sys.path.insert(0, str(local_ext))

if importlib.util.find_spec("cat_envs") is None:
    print(
        "[ERROR] Missing local package `cat_envs`.\n"
        "Either install it or run from the repo with the local path available.\n"
        "Tried adding: exts/cat_envs"
    )
    raise SystemExit(1)

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import collections
import importlib
import os
import re
import torch
import omni.kit.app
import yaml

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    export_policy_as_jit,
    export_policy_as_onnx,
)

import cat_envs.tasks  # noqa: F401
from cat_envs.tasks.utils.rsl_rl.rsl_rl import RslRlVecEnvWrapperCaT

# Enable debug draw extension for optional visualizers.
try:
    ext_manager = omni.kit.app.get_app().get_extension_manager()
    try:
        ext_manager.set_extension_enabled_immediate("isaacsim.util.debug_draw", True)
    except Exception:
        ext_manager.set_extension_enabled_immediate("omni.isaac.debug_draw", True)
except Exception:
    pass


def load_cfg_from_registry(task_name: str, entry_point_key: str) -> dict | object:
    """Load default configuration given its entry point from the gym registry."""
    cfg_entry_point = gym.spec(task_name.split(":")[-1]).kwargs.get(entry_point_key)
    if cfg_entry_point is None:
        agents = collections.defaultdict(list)
        for k in gym.spec(task_name.split(":")[-1]).kwargs:
            if k.endswith("_cfg_entry_point") and k != "env_cfg_entry_point":
                spec = (
                    k.replace("_cfg_entry_point", "")
                    .replace("rl_games", "rl-games")
                    .replace("rsl_rl", "rsl-rl")
                    .split("_")
                )
                agent = spec[0].replace("-", "_")
                algorithms = [item.upper() for item in (spec[1:] if len(spec) > 1 else ["PPO"])]
                agents[agent].extend(algorithms)
        msg = "\nExisting RL library (and algorithms) config entry points: "
        for agent, algorithms in agents.items():
            msg += f"\n  |-- {agent}: {', '.join(algorithms)}"
        raise ValueError(
            f"Could not find configuration for the environment: '{task_name}'."
            f"\nPlease check that the gym registry has the entry point: '{entry_point_key}'."
            f"{msg if agents else ''}"
        )
    if isinstance(cfg_entry_point, str) and cfg_entry_point.endswith(".yaml"):
        if os.path.exists(cfg_entry_point):
            config_file = cfg_entry_point
        else:
            mod_name, file_name = cfg_entry_point.split(":")
            mod_path = os.path.dirname(importlib.import_module(mod_name).__file__)
            config_file = os.path.join(mod_path, file_name)
        print(f"[INFO]: Parsing configuration from: {config_file}")
        with open(config_file, encoding="utf-8") as f:
            cfg = yaml.full_load(f)
    else:
        if callable(cfg_entry_point):
            cfg_cls = cfg_entry_point()
        elif isinstance(cfg_entry_point, str):
            mod_name, attr_name = cfg_entry_point.split(":")
            mod = importlib.import_module(mod_name)
            cfg_cls = getattr(mod, attr_name)
        else:
            cfg_cls = cfg_entry_point
        print(f"[INFO]: Parsing configuration from: {cfg_entry_point}")
        cfg = cfg_cls() if callable(cfg_cls) else cfg_cls
    return cfg


def parse_env_cfg(
    task_name: str, device: str = "cuda:0", num_envs: int | None = None, use_fabric: bool | None = None
) -> ManagerBasedRLEnvCfg | DirectRLEnvCfg:
    """Parse configuration for an environment and override based on inputs."""
    cfg = load_cfg_from_registry(task_name.split(":")[-1], "env_cfg_entry_point")
    if isinstance(cfg, dict):
        raise RuntimeError(f"Configuration for the task: '{task_name}' is not a class. Please provide a class.")
    cfg.sim.device = device
    if use_fabric is not None:
        cfg.sim.use_fabric = use_fabric
    if num_envs is not None:
        cfg.scene.num_envs = num_envs
    return cfg


def get_checkpoint_path(
    log_path: str, run_dir: str = ".*", checkpoint: str = ".*", other_dirs: list[str] = None, sort_alpha: bool = True
) -> str:
    """Get path to the model checkpoint in input directory."""
    try:
        runs = [
            os.path.join(log_path, run) for run in os.scandir(log_path) if run.is_dir() and re.match(run_dir, run.name)
        ]
        if sort_alpha:
            runs.sort()
        else:
            runs = sorted(runs, key=os.path.getmtime)
        run_path = runs[-1]
    except Exception:
        raise ValueError(f"No runs found in: {log_path}")

    if other_dirs is not None:
        for dir_name in other_dirs:
            run_path = os.path.join(run_path, dir_name)

    try:
        checkpoints = [
            os.path.join(run_path, file)
            for file in os.scandir(run_path)
            if file.is_file() and re.match(checkpoint, file.name)
        ]
        if sort_alpha:
            checkpoints.sort()
        else:
            checkpoints = sorted(checkpoints, key=os.path.getmtime)
        checkpoint_path = checkpoints[-1]
    except Exception:
        raise ValueError(f"No checkpoints found in: {run_path}")

    return checkpoint_path


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    if hasattr(env_cfg, "viewer"):
        env_cfg.viewer.resolution = (1280, 720)
    if hasattr(env_cfg, "sim") and hasattr(env_cfg, "decimation"):
        # Render once per control step to reduce viewport overhead.
        env_cfg.sim.render_interval = max(1, env_cfg.decimation)
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(
        args_cli.task, args_cli
    )
    use_pretrained = args_cli.pretrained or args_cli.pretrained_path is not None
    pretrained_path = None

    if args_cli.center:
        env_cfg.viewer.origin_type = "asset_root"
        env_cfg.viewer.asset_name = "robot"
        env_cfg.viewer.env_index = 10
        env_cfg.viewer.eye = (3.0, 3.0, 3.0)
        env_cfg.viewer.resolution = (1280, 720)
        if args_cli.num_envs is not None and args_cli.num_envs > 0:
            env_cfg.viewer.env_index = min(
                env_cfg.viewer.env_index, args_cli.num_envs - 1
            )

    if use_pretrained:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        default_pretrained_path = os.path.normpath(
            os.path.join(
                script_dir,
                "..",
                "..",
                "assets",
                "pretrained",
                "policy.pt",
            )
        )
        pretrained_path = os.path.abspath(args_cli.pretrained_path or default_pretrained_path)
        if not os.path.isfile(pretrained_path):
            raise FileNotFoundError(
                f"Pretrained policy not found at: {pretrained_path}. "
                "Use --pretrained_path to point to a TorchScript policy."
            )
        log_dir = os.path.abspath(os.path.join("logs", "rsl_rl", "pretrained_play"))
    else:
        # specify directory for logging experiments
        log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        log_root_path = os.path.abspath(log_root_path)
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        resume_path = get_checkpoint_path(
            log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint
        )
        log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(
        args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None
    )
    # pass steps-per-iteration for iteration-based curriculums
    if hasattr(env, "unwrapped"):
        setattr(env.unwrapped, "_rsl_num_steps_per_env", agent_cfg.num_steps_per_env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    clip_actions = getattr(agent_cfg, "clip_actions", None)
    env = RslRlVecEnvWrapperCaT(env, clip_actions=clip_actions)
    runner_cfg = agent_cfg.to_dict()
    available_obs = set()
    try:
        obs_preview = env.get_observations()
        if hasattr(obs_preview, "keys"):
            available_obs = {str(k) for k in obs_preview.keys()}
    except Exception:
        pass
    obs_groups = runner_cfg.get("obs_groups")
    if isinstance(obs_groups, dict) and "policy" in available_obs:
        critic_groups = obs_groups.get("critic")
        if isinstance(critic_groups, list) and any(group not in available_obs for group in critic_groups):
            print(
                f"[INFO] Falling back critic obs_groups from {critic_groups} to ['policy'] "
                f"(available: {sorted(available_obs)})"
            )
            obs_groups["critic"] = ["policy"]
            runner_cfg["obs_groups"] = obs_groups

    if use_pretrained:
        print(f"[INFO]: Loading pretrained policy from: {pretrained_path}")
        policy = torch.jit.load(pretrained_path, map_location=env.unwrapped.device)
        policy.eval()
        if hasattr(policy, "reset"):
            policy.reset()
    else:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        ppo_runner = OnPolicyRunner(
            env, runner_cfg, log_dir=None, device=agent_cfg.device
        )
        ppo_runner.load(resume_path)

        # obtain the trained policy for inference
        policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

        # extract the neural network module
        # we do this in a try-except to maintain backwards compatibility.
        try:
            # version 2.3 onwards
            policy_nn = ppo_runner.alg.policy
        except AttributeError:
            # version 2.2 and below
            policy_nn = ppo_runner.alg.actor_critic

        # extract the normalizer
        if hasattr(policy_nn, "actor_obs_normalizer"):
            normalizer = policy_nn.actor_obs_normalizer
        elif hasattr(policy_nn, "student_obs_normalizer"):
            normalizer = policy_nn.student_obs_normalizer
        else:
            normalizer = None

        # export policy to onnx/jit
        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
        export_policy_as_jit(
            policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt"
        )
        export_policy_as_onnx(
            policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx"
        )

    # reset environment
    obs = env.get_observations()
    if isinstance(obs, tuple):
        obs = obs[0]
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            policy_obs = obs
            if use_pretrained and hasattr(obs, "get"):
                policy_obs = obs.get("policy", obs)
            actions = policy(policy_obs)
            # env stepping
            obs, _, _, _ = env.step(actions)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
