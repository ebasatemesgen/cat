# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from cat_envs.tasks.utils.cat.manager_constraint_cfg import (
    ConstraintTermCfg as ConstraintTerm,
)
import cat_envs.tasks.utils.cat.constraints as constraints

from .cat_flat_env_cfg import ConstraintsCfg, Solo12FlatEnvCfg

LIMBO_HEIGHT_M = 0.25
LIMBO_BAR_THICKNESS_M = 0.05

LIMBO_BAR_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/LimboBar",
    spawn=sim_utils.CuboidCfg(
        size=(0.2, 2.0, LIMBO_BAR_THICKNESS_M),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            kinematic_enabled=True,
            disable_gravity=True,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
            contact_offset=0.02,
            rest_offset=0.0,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.3, 0.3)),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(1.0, 0.0, LIMBO_HEIGHT_M + 0.5 * LIMBO_BAR_THICKNESS_M),
    ),
)


@configclass
class LimboConstraintsCfg(ConstraintsCfg):
    """Adds a max base height constraint for limbo locomotion."""

    max_base_height = ConstraintTerm(
        func=constraints.max_base_height,
        max_p=1.0,
        params={"limit": LIMBO_HEIGHT_M, "asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class Solo12LimboEnvCfg(Solo12FlatEnvCfg):
    """Limbo environment with a low overhead bar."""

    constraints: LimboConstraintsCfg = LimboConstraintsCfg()

    def __post_init__(self):
        super().__post_init__()
        # add limbo bar to the scene
        self.scene.limbo_bar = LIMBO_BAR_CFG


@configclass
class Solo12LimboEnvCfg_PLAY(Solo12LimboEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        # smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 3.0

        # disable randomization for play
        self.observations.policy.enable_corruption = False
