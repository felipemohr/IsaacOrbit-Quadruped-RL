import gymnasium as gym

from omni.isaac.orbit_quadruped_rl import quadruped_env_cfg
from . import agents

gym.register(
    id="Isaac-Quadruped-Go2-v0",
    entry_point="omni.isaac.orbit.envs:RLTaskEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": quadruped_env_cfg.QuadrupedEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.QuadrupedGo2PPORunnerCfg,
    },
)
