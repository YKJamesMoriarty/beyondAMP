from isaaclab.utils import configclass
from isaaclab.managers import EventTermCfg as EventTerm

from robotlib.beyondMimic.robots.g1 import G1_ACTION_SCALE, G1_CYLINDER_CFG

from amp_tasks.others.amp_env_cfg import AMPEnvCfg, EventCfg
from beyondAMP.obs_groups import AMPObsBaiscCfg
import beyondAMP.mdp as mdp


@configclass
class G1StanceEventsCfg(EventCfg):
    """Stance 任务事件配置。

    设计目标：
    1) 每次 reset 都从 demo 中随机采样一帧状态重置机器人；
    2) 不额外叠加 reset 噪声，确保初始化语义清晰；
    3) 关闭 interval 推搡干扰，避免纯风格训练被外力扰动主导。
    """

    reset_to_ref_motion_dataset = EventTerm(
        func=mdp.reset_to_ref_motion_dataset,
        mode="reset",
        params={
            # 根位姿偏移噪声（这里为 0，表示严格使用 demo 采样状态）
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0), "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)},
            # 根速度偏移噪声（注意 key 使用 vx/vy/vz/wx/wy/wz）
            "velocity_range": {
                "vx": (0.0, 0.0),
                "vy": (0.0, 0.0),
                "vz": (0.0, 0.0),
                "wx": (0.0, 0.0),
                "wy": (0.0, 0.0),
                "wz": (0.0, 0.0),
            },
            # 关节位置/速度偏移噪声（0 表示不扰动）
            "joint_position_range": (0.0, 0.0),
            "joint_velocity_range": (0.0, 0.0),
        },
    )

    def __post_init__(self):
        super().__post_init__()
        # stance 纯 AMP 训练中不使用 interval 外力推搡。
        self.push_robot = None


@configclass
class G1StanceFlatEnvCfg(AMPEnvCfg):
    """G1 stance 动作的 AMP-only 环境配置。"""

    events = G1StanceEventsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = G1_ACTION_SCALE
        # 显式指定 AMP 观测组，避免基类改动带来的行为漂移。
        self.observations.amp = AMPObsBaiscCfg()
