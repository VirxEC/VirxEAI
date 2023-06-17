import os

import torch
from stable_baselines3 import PPO


class Agent:
    def __init__(self):
        base_folder = os.path.dirname(os.path.realpath(__file__))
        self.actor = PPO.load(os.path.join(base_folder, "VirxEAI.zip"))

    def act(self, state):
        with torch.no_grad():
            return self.actor(state)[0]
