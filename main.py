import json
import math
import os
import random
from dataclasses import dataclass
from datetime import datetime
from time import time_ns
from typing import List, Tuple

import numpy as np
import rlgym
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition
from torch.utils import tensorboard

from making import Obs, Reward
from utils import *

TRAIN = True

tick_skip = 8
max_steps = int(round(60 * 120 / tick_skip))

#All we have to do now is pass our custom configuration objects to rlgym!
num_players = 2
env = rlgym.make(
    "default self",
    random_resets=True,
    team_size=int(round(num_players / 2)),
    tick_skip=tick_skip,
    reward_fn=Reward(),
    obs_builder=Obs(),
    terminal_conditions=[TimeoutCondition(max_steps),]
)

s = 0
new_ai = True

base_folder = os.path.dirname(os.path.realpath(__file__))
w = [tensorboard.SummaryWriter(log_dir=os.path.join(base_folder, "runs", f"{i}-{datetime.now().strftime('%Y-%m-%d %H;%M')}")) for i in range(num_players)]
player = Player(num_players, new_ai, base_folder, TRAIN)

while True:
    mspt = []
    done = False
    new_ai = False
    state = env.reset()

    while not done:
        # start the tick
        start = time_ns()

        # get the action and give it to rlgym
        actions = player.step(state)
        state, reward, done, gameinfo = env.step(actions)

        # give the model the rewards it got
        player.add_reward(reward)

        # end the tick
        end = time_ns()
        mspt.append((end - start) / 1_000_000)
        if mspt[-1] > 100:
            print(f"Tick took {mspt[-1]}ms")

    print(f"Episode {s} finished with net rewards of {player.get_total_rewards()} and an average mspt of {sum(mspt) / len(mspt)}")

    print(f"Training model...")
    start = time_ns()

    # train the model
    player.learn(w, s)

    ms = (time_ns() - start) / 1_000_000
    print(f"Trained in {ms}ms. Saving...")

    # end the episode
    player.end_episode(w, s)
    s += 1
    mspt = []

    print(f"Saved. Continue with the episode {s}")
