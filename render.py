import time
import numpy as np
import rlgym_sim
from rlgym_sim.gym import Gym
from rlgym_sim.utils.state_setters import DefaultState
from rlgym_sim.utils.terminal_conditions.common_conditions import (
    GoalScoredCondition, TimeoutCondition)

from agent import Agent
from making import REWARD, Obs
from rlgym_tools.extra_action_parsers.kbm_act import KBMAction

TPS = 15


def main():
    print("Running match...")

    env: Gym = rlgym_sim.make(
        reward_fn=REWARD,
        terminal_conditions=[GoalScoredCondition(), TimeoutCondition(300)],
        obs_builder=Obs(),
        action_parser=KBMAction(),
        state_setter=DefaultState(),
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