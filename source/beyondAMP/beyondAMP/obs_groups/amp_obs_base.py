from __future__ import annotations

from isaaclab.utils import configclass

from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm

import beyondAMP.mdp as mdp

"""
Not that you should update the 
"""
@configclass
class AMPObsBaseCfg(ObsGroup):
    def adjust_key_body_indexes(self, terms:list, key_bodys:list):
        for term_name in terms:
            term:ObsTerm = getattr(self, term_name)
            if "asset_cfg" in term.params:
                term.params["asset_cfg"].body_names = key_bodys
            else:
                term.params["asset_cfg"] = SceneEntityCfg(name="robot", body_names=key_bodys)
        return self

@configclass
class AMPObsBaiscCfg(AMPObsBaseCfg):
    joint_pos = ObsTerm(func=mdp.joint_pos_rel)
    joint_vel = ObsTerm(func=mdp.joint_vel_rel)
    
AMPObsBaiscTerms = ["joint_pos", "joint_vel"]

@configclass
class AMPObsClassicCfg(AMPObsBaseCfg):
    joint_pos = ObsTerm(func=mdp.joint_pos_rel)
    joint_vel = ObsTerm(func=mdp.joint_vel_rel)
    base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
    
AMPObsClassicTerms = ["joint_pos", "joint_vel", "base_lin_vel", "base_ang_vel"]


@configclass
class AMPObsG1MimicRootKeyCfg(AMPObsBaseCfg):
    """G1 风格判别器观测（贴近 MimicKit AMP 设计）。

    单帧包含：
    1) joint_pos_rel
    2) joint_vel_rel
    3) root_pos_w（root 位置，后续在历史窗口内做 x/y 相对化）
    4) body_pos_w（key body 世界坐标，后续转为相对 root）
    """

    joint_pos = ObsTerm(func=mdp.joint_pos_rel)
    joint_vel = ObsTerm(func=mdp.joint_vel_rel)
    root_pos_w = ObsTerm(func=mdp.root_pos_w)
    # 默认 key body 选 MimicKit G1 stance 的 5 个关键点。
    body_pos_w = ObsTerm(
        func=mdp.body_pos_w,
        params={
            "asset_cfg": SceneEntityCfg(
                name="robot",
                body_names=[
                    "left_ankle_roll_link",
                    "right_ankle_roll_link",
                    "torso_link",
                    "left_wrist_yaw_link",
                    "right_wrist_yaw_link",
                ],
            )
        },
    )


AMPObsG1MimicRootKeyTerms = ["joint_pos", "joint_vel", "root_pos_w", "body_pos_w"]
