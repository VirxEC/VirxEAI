import time
from pathlib import Path

import numpy as np
import rlgym_sim
from rlgym_sim.gym import Gym

from agent import Agent
from making import REWARD, TERMINAL, Obs, ReplaySetter
from rlgym_tools.extra_action_parsers.kbm_act import KBMAction

TPS = 15


def main():
    print("Running match...")

    with open("replay_path.txt", "r") as f:
        replays_path = Path(f.read().strip()) / "ranked-duels"

    replays = list(replays_path.glob("*.bin"))
    num_replays = len(replays)

    env: Gym = rlgym_sim.make(
        reward_fn=REWARD,
        terminal_conditions=TERMINAL,
        obs_builder=Obs(),
        action_parser=KBMAction(),
        state_setter=ReplaySetter(replays, num_replays),
        team_size=1,
        spawn_opponents=True,
    )

    try:
        while True:
            agents: list[Agent] = [Agent() for _ in range(env._match.agents)]
            obs = env.reset()
            done = False

            steps = 0
            starttime = time.time()

            while not done:
                actions = [agents[i].act(obs[i]) for i in range(env._match.agents)]

                new_obs, _, done, _ = env.step(np.array(actions))
                # new_obs, _, done, _ = env.step(np.array(actions), dt)

                env.render()

                obs = new_obs

                # Sleep to keep the game in real time
                steps += 1
                time.sleep(max(0, starttime + steps / TPS - time.time()))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()