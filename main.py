import os
from pathlib import Path

from rlgym.utils.terminal_conditions.common_conditions import \
    GoalScoredCondition
from rlgym.wrappers.sb3_wrappers import SB3MultipleInstanceWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor

from making import Obs, Reward

if __name__ == "__main__":
    NEW_AI = False

    path_to_rl="C:\\Program Files\\Epic Games\\rocketleague\\Binaries\\Win64\\rocketleague.exe"

    def get_args():
        return dict(
            self_play=True,
            team_size=1,
            random_resets=True,
            reward_function=Reward(),
            obs_builder=Obs(),
            terminal_conditions=[GoalScoredCondition(),],
        )

    env = VecMonitor(SB3MultipleInstanceWrapper(path_to_rl, 3, get_args, wait_time=20))

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
            model.learn(100_000_000, callback=CheckpointCallback(100_000, "VirxEAI"))
        except KeyboardInterrupt:
            model.save("VirxEAI")
            break
