from stable_baselines3 import PPO
import os

class Agent:
    def __init__(self):
        base_folder = os.path.dirname(os.path.realpath(__file__))
        self.actor = PPO.load(os.path.join(base_folder, "VirxEAI.zip"))

    def act(self, state):
        # Evaluate your model here
        return self.actor.predict(state)[0]
