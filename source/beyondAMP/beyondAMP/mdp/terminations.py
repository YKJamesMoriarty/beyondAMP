from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg


def robot_falling(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    height_threshold: float = 0.25,
    tilt_threshold: float = 0.695,
) -> torch.Tensor:
    """检测机器人是否摔倒（高度过低或姿态过度倾斜）。

    逻辑参考 whole_body_tracking 项目中的 ``robot_falling``：
    1) root 高度低于阈值；
    2) 机器人基座局部 z 轴与世界 z 轴夹角过大（通过点积阈值判断）。
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # 条件1：根节点高度过低。
    root_height = asset.data.root_pos_w[:, 2]
    height_too_low = root_height < height_threshold

    # 条件2：机体过度倾斜（局部 z 轴在世界系的 z 分量过小）。
    root_quat_w = asset.data.root_quat_w
    local_up = torch.zeros((env.num_envs, 3), device=env.device)
    local_up[:, 2] = 1.0
    world_up_body = math_utils.quat_apply(root_quat_w, local_up)
    too_tilted = world_up_body[:, 2] < tilt_threshold

    return height_too_low | too_tilted
