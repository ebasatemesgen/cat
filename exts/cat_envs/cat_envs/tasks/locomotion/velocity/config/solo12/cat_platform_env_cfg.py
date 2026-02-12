# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

from .cat_flat_env_cfg import MySceneCfg, Solo12FlatEnvCfg

# Terrain mix to approximate the "platform / stairs / slopes / obstacles" scenarios.
PLATFORM_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(6.0, 6.0),
    border_width=0.5,
    num_rows=4,
    num_cols=4,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "platform_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.5,
            step_height_range=(0.04, 0.06),
            step_width=0.4,
            platform_width=2.0,
            border_width=0.5,
            holes=False,
        ),
        "platform_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.3,
            slope_range=(0.0, 0.35),
            platform_width=2.0,
            border_width=0.5,
        ),
        "platform_boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.2,
            # grid_width must not evenly divide size (6.0) so border_width > 0
            grid_width=0.45,
            grid_height_range=(0.05, 0.2),
            platform_width=2.0,
        ),
    },
)


@configclass
class PlatformSceneCfg(MySceneCfg):
    """Scene with rough terrain (stairs/slopes/obstacles)."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=PLATFORM_TERRAINS_CFG,
        max_init_terrain_level=3,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )


@configclass
class Solo12PlatformEnvCfg(Solo12FlatEnvCfg):
    """Platform/stairs/slopes environment for Solo12."""

    scene: PlatformSceneCfg = PlatformSceneCfg(num_envs=4096, env_spacing=4.0)

    def __post_init__(self):
        super().__post_init__()
        # Ensure terrain uses scene physics material
        self.sim.physics_material = self.scene.terrain.physics_material


@configclass
class Solo12PlatformEnvCfg_PLAY(Solo12PlatformEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        # smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 4.0

        # disable randomization for play
        self.observations.policy.enable_corruption = False
