from typing import List
from isaaclab.utils import configclass
from .rl_cfg import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

from beyondAMP.obs_groups import AMPObsBaiscCfg
from rsl_rl_amp.runners.amp_on_policy_runner import AMPOnPolicyRunner
from beyondAMP.motion.motion_dataset import MotionDatasetCfg
from dataclasses import MISSING

@configclass
class AMPPPOAlgorithmCfg(RslRlPpoAlgorithmCfg):    
    class_name="AMPPPO"
    amp_replay_buffer_size: int = 100000
    # 判别器 trunk 与 head 的 L2 正则（weight decay）。
    # 默认值保持与旧实现一致，避免影响现有任务行为。
    amp_trunk_weight_decay: float = 1e-3
    amp_head_weight_decay: float = 1e-1

@configclass
class AMPPPOWeightedAlgorithmCfg(AMPPPOAlgorithmCfg):    
    class_name="AMPPPOWeighted"
    rescore_interval: int = 50

@configclass
class AMPRunnerCfg(RslRlOnPolicyRunnerCfg):
    runner_type:             type[AMPOnPolicyRunner] = AMPOnPolicyRunner
    amp_data:               MotionDatasetCfg = MISSING
    amp_reward_coef:        float = MISSING
    amp_discr_hidden_dims:  List[int] = MISSING
    # The coef to consider the task, 1 to only consider task and 0 to only consider amp
    amp_task_reward_lerp:   float = 0.9
    amp_min_normalized_std: float = 0.0 # recmended for no minimal explore std. since the action limit may large 
