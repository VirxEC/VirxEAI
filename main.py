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
from rlgym.utils.reward_functions.common_rewards import MoveTowardsBallReward
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition
from torch.utils import tensorboard

from making import Obs
from utils import *

TRAIN = True

tick_skip = 8
max_steps = int(round(10 * 120 / tick_skip))

#All we have to do now is pass our custom configuration objects to rlgym!
num_players = 2
env = rlgym.make(
    "default self",
    random_resets=True,
    team_size=int(round(num_players / 2)),
    tick_skip=tick_skip,
    reward_fn=MoveTowardsBallReward(),
    obs_builder=Obs(),
    terminal_conditions=[TimeoutCondition(max_steps),]
)

s = 0
num_episodes = 0
new_ai = True

base_folder = os.path.dirname(os.path.realpath(__file__))
w = [tensorboard.SummaryWriter(log_dir=os.path.join(base_folder, "runs", f"{i}-{datetime.now().strftime('%Y-%m-%d %H;%M')}")) for i in range(num_players)]
player = Player(new_ai, base_folder, TRAIN)

while True:
    mspt = []
    done = False
    new_ai = False
    state = env.reset()

    while not done:
        start = time_ns()
        actions = player.step(num_players, state)
        next_state, reward, done, gameinfo = env.step(actions)

        player.learn(num_players, next_state, state, reward, w, s)

        state = [st.clone() for st in next_state]
        end = time_ns()
        mspt.append((end - start) / 1_000_000)

    print(f"Episode {num_episodes} finished with net rewards of {player.total_reward} and an average mspt of {sum(mspt) / len(mspt)}")
    player.end_episode(num_players, w, num_episodes)

    num_episodes += 1
    mspt = []
