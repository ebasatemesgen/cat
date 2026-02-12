# Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from __future__ import annotations

import torch
from tensordict import TensorDict

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper


class RslRlVecEnvWrapperCaT(RslRlVecEnvWrapper):
    """CaT-aware RSL-RL wrapper.

    RSL-RL's default wrapper converts done signals to binary values
    `(terminated | truncated)`. For CaT we keep float done values so PPO uses
    soft termination probabilities in return computation.
    """

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        # clip actions if requested
        if self.clip_actions is not None:
            actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)

        obs_dict, rew, terminated, truncated, extras = self.env.step(actions)

        # keep CaT float done values (do not collapse to binary done)
        dones = terminated.to(dtype=torch.float32)
        if dones.dim() > 1:
            dones = dones.squeeze(-1)

        if rew.dim() > 1:
            rew = rew.squeeze(-1)

        # keep timeout information for reward bootstrapping
        if not self.unwrapped.cfg.is_finite_horizon:
            extras["time_outs"] = truncated

        return TensorDict(obs_dict, batch_size=[self.num_envs]), rew, dones, extras
