import os
from pathlib import Path

import torch
from rlgym_sim.envs.match import Match
from rlgym_sim.utils.state_setters import DefaultState
from rlgym_sim.utils.terminal_conditions.common_conditions import (
    GoalScoredCondition, TimeoutCondition)
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor

from making import REWARD, Obs
from rlgym_tools.extra_action_parsers.kbm_act import KBMAction
from rlgym_tools.sb3_tools.sb3_multiple_instance_env import \
    SB3MultipleInstanceEnv

if __name__ == "__main__":
    NEW_AI = True

    def get_match():
        return Match(
            reward_function=REWARD,
            terminal_conditions=[GoalScoredCondition(), TimeoutCondition(300)],
            obs_builder=Obs(),
            action_parser=KBMAction(),
            state_setter=DefaultState(),
            team_size=1,
            spawn_opponents=True,
        )

    env = VecMonitor(SB3MultipleInstanceEnv(get_match, "auto"))

    base_folder = Path(os.path.dirname(os.path.realpath(__file__)))
    logs_folder = base_folder / "logs"

    if not os.path.isdir(logs_folder):
        os.mkdir(logs_folder)

    # Initialize PPO from stable_baselines3

    ppo_args = {
        "env": env,
        "verbose": 1,
        "tensorboard_log": logs_folder
    }

    model = PPO("MlpPolicy", **ppo_args) if NEW_AI else PPO.load(base_folder / "VirxEAI.zip", **ppo_args)

    # Train our agent!
    while True:
        try:
            print("Training model...")
            model.learn(100_000_000, callback=CheckpointCallback(100_000, "VirxEAI"))
        except KeyboardInterrupt:
            model.save("VirxEAI")
            break
