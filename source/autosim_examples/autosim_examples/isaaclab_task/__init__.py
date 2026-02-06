import gymnasium as gym

gym.register(
    id="AutoSimExamples-IsaacLab-FrankaCubeLift-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_lift_cube_cfg:FrankaCubeLiftEnvCfg",
    },
    disable_env_checker=True,
)
