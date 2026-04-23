from isaaclab.utils import configclass
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import mdp

from robotlib.beyondMimic.robots.g1 import G1_ACTION_SCALE, G1_CYLINDER_CFG

from amp_tasks.others.amp_env_cfg import AMPEnvCfg, EventCfg
from beyondAMP.obs_groups import AMPObsG1MimicRootKeyCfg
from .agents import general


def action_rate_l2_scaled(env, scale: float = 0.1):
    """对 action_rate_l2 做额外缩放，避免早期负惩罚过强。"""
    return scale * mdp.action_rate_l2(env)


@configclass
class G1StanceRewardsCfg:
    """Stance 任务专用奖励：只保留最小稳定项。"""

    # 存活奖励：每个 step 未终止则给 1（再乘以权重）。
    alive = RewTerm(func=mdp.is_alive, weight=2.0)
    # 动作变化率惩罚：在原公式上额外乘 0.1，再乘 term 权重。
    # 原始公式: sum((a_t - a_{t-1})^2)
    action_rate_l2 = RewTerm(func=action_rate_l2_scaled, params={"scale": 0.1}, weight=-1e-1)


@configclass
class G1StanceEventsCfg(EventCfg):
    """Stance 任务事件配置。

    设计目标：
    1) reset 行为与 punch hit 对齐：从“默认站立附近 + 随机扰动”启动；
    2) 不依赖 demo 帧做初始化，更贴近 sim2real 上电/复位流程；
    3) 关闭 interval 推搡干扰，避免纯风格训练被外力扰动主导。
    """

    # sim2real 友好的 root 初始化：
    # - 在默认落点附近做小范围 x/y 扰动；
    # - yaw 小范围扰动，避免初始朝向单一但不过度困难；
    # - 速度清零。
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3), "yaw": (-0.4, 0.4)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )
    # 对齐作者原生任务：关节 reset 使用 scale 随机化（基于默认关节状态做比例缩放）。
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (-1.0, 1.0),
            "velocity_range": (0.5, 1.0),
        },
    )

    def __post_init__(self):
        super().__post_init__()
        # stance 不再使用“从 demo 随机帧重置”。
        self.reset_to_ref_motion_dataset = None
        # stance 纯 AMP 训练中不使用 interval 外力推搡。
        self.push_robot = None


@configclass
class G1StanceFlatEnvCfg(AMPEnvCfg):
    """G1 stance 动作的 AMP-only 环境配置。"""

    events = G1StanceEventsCfg()
    rewards = G1StanceRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = G1_ACTION_SCALE
        # 使用 MimicKit 风格 AMP 观测：joint + root_pos + key_body_pos。
        self.observations.amp = AMPObsG1MimicRootKeyCfg()
        # 与 stance 任务配置保持一致：key body 明确从 general 读取，避免硬编码漂移。
        self.observations.amp.body_pos_w.params["asset_cfg"] = SceneEntityCfg(
            name="robot", body_names=general.key_body_names
        )
