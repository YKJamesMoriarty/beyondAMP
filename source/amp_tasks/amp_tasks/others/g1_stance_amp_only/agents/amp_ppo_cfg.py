from isaaclab.utils import configclass

from beyondAMP.isaaclab.rsl_rl.configs.rl_cfg import RslRlPpoActorCriticCfg
from beyondAMP.isaaclab.rsl_rl.configs.amp_cfg import (
    AMPPPOWeightedAlgorithmCfg,
    AMPRunnerCfg,
    MotionDatasetCfg,
)
from beyondAMP.obs_groups import AMPObsG1MimicRootKeyTerms

from . import general


@configclass
class G1StanceAMPRunnerCfg(AMPRunnerCfg):
    num_steps_per_env = 24
    # stance 动作相对细腻，默认训练轮数提高，避免 1w 轮过早停止。
    max_iterations = 50000
    # 与 punch_hit 任务保持一致：每 500 iter 保存一次模型。
    save_interval = 500
    experiment_name = general.experiment_name
    run_name = "amp_pure"
    empirical_normalization = True

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = AMPPPOWeightedAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        # 历史基线：未拆分学习率时，actor/critic/disc 共用 learning_rate=1e-3。
        learning_rate=1.0e-3,
        # 分离学习率：policy/critic 快一些，disc 慢一些。
        # 若不显式设置 policy_learning_rate / disc_learning_rate，则都会回退到上面的 learning_rate。
        policy_learning_rate=1.0e-3,
        disc_learning_rate=1.0e-3,
        # actor+critic 更新 2 次，disc 更新 1 次（减缓判别器过快收敛）。
        # 原始数值为1
        disc_update_interval=1,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        rescore_interval=100,
        # 让判别器收敛更慢：提高判别器参数 L2 正则。
        # amp_trunk_weight_decay=1e-3,
        # amp_head_weight_decay=1e-1,
        amp_trunk_weight_decay=1e-3,
        amp_head_weight_decay=1e-1,
    )

    amp_data = MotionDatasetCfg(
        motion_files=general.amp_data_files,
        body_names=general.key_body_names,
        anchor_name=general.anchor_name,
        root_name="pelvis",
        amp_obs_terms=AMPObsG1MimicRootKeyTerms,
        # MimicKit 风格：判别器使用 10 帧历史窗口。
        history_steps=10,
        # root position: x/y 相对历史窗口最新帧；z 保持绝对值。
        root_xy_pos_rel_to_latest=True,
        # key body position: 每帧都转为相对于同帧 root 的位置。
        key_body_pos_rel_to_root=True,
    )
    amp_discr_hidden_dims = [256, 256]
    amp_reward_coef = 1.0
    # 纯 AMP：PPO 更新只使用判别器生成的风格奖励，不混入任务奖励。
    amp_task_reward_lerp = 0.0
