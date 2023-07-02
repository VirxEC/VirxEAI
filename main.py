import os
from pathlib import Path

import torch
from rlgym_sim.envs.match import Match
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor

from making import REWARD, TERMINAL, Obs, ReplaySetter
from rlgym_tools.extra_action_parsers.kbm_act import KBMAction
from rlgym_tools.sb3_tools.sb3_multiple_instance_env import \
    SB3MultipleInstanceEnv

if __name__ == "__main__":
    NEW_AI = False
    GPU_ACCEL = False

    # check if "libpt_ocl.so" exists in the current directory
    if GPU_ACCEL and os.path.isfile("libpt_ocl.so"):
        torch.ops.load_library("libpt_ocl.so")
        device = "privateuseone:0"
    else:
        device = "auto"

    with open("replay_path.txt", "r") as f:
        replays_path = Path(f.read().strip()) / "ranked-duels"

    replays = list(replays_path.glob("*.bin"))
    num_replays = len(replays)

    def get_match():
        return Match(
            reward_function=REWARD,
            terminal_conditions=TERMINAL,
            obs_builder=Obs(),
            action_parser=KBMAction(),
            state_setter=ReplaySetter(replays, num_replays),
            team_size=1,
            spawn_opponents=True,
        )

    online = True

    while online:
        print("Initiating parallel model training environments...")
        env = VecMonitor(SB3MultipleInstanceEnv(get_match, "auto"))

        base_folder = Path(os.path.dirname(os.path.realpath(__file__)))
        logs_folder = base_folder / "logs"

        if not os.path.isdir(logs_folder):
            os.mkdir(logs_folder)

        ppo_args = {
            "env": env,
            "verbose": 1,
            "tensorboard_log": logs_folder,
            "device": device,
        }

        model = PPO("MlpPolicy", **ppo_args) if NEW_AI else PPO.load(base_folder / "VirxEAI.zip", **ppo_args)
        NEW_AI = False

        try:
            # Train our agent!
            while online:
                try:
                    print("Training model...")
                    model.learn(1_000_000_000, callback=CheckpointCallback(20_000, "VirxEAI", "VirxEAI"))
                except InterruptedError:
                    model.save("VirxEAI")
                    online = False
        except Exception as e:
            model.save("VirxEAI")
            print(e)
            # online = False

