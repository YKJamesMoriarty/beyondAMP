import gymnasium as gym

from . import g1_stance_amp_only_env_cfg
from .agents import amp_ppo_cfg, base_ppo_cfg

##
# Register Gym environments.
##

gym.register(
    id="beyondAMP-StanceTask-G1-PPO",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": g1_stance_amp_only_env_cfg.G1StanceFlatEnvCfg,
        "rsl_rl_cfg_entry_point": base_ppo_cfg.G1StancePPORunnerCfg,
    },
)

gym.register(
    id="beyondAMP-StanceTask-G1-AMPBasic",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": g1_stance_amp_only_env_cfg.G1StanceFlatEnvCfg,
        "rsl_rl_cfg_entry_point": amp_ppo_cfg.G1StanceAMPRunnerCfg,
    },
)
