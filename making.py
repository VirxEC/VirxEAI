import math
from typing import Any, List

import numpy as np
from rlgym.utils import common_values
from rlgym.utils import math as rlmath
from rlgym.utils.gamestates import GameState, PhysicsObject, PlayerData
from rlgym.utils.obs_builders import ObsBuilder
from rlgym.utils.reward_functions import RewardFunction

from utils import t


class Reward(RewardFunction):
    def reset(self, initial_state: GameState, optional_data=None):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray, optional_data=None):
        # Vector version of v=d/t <=> t=d/v <=> 1/t=v/d
        # Max value should be max_speed / ball_radius = 2300 / 94 = 24.5 / 100 = 0.245
        # Used to guide the agent towards the ball
        inv_t = rlmath.scalar_projection(player.car_data.linear_velocity, state.ball.position - player.car_data.position)
        return inv_t / 100

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray, optional_data=None):
        return 0


class Obs(ObsBuilder):
    POS_STD = 2300
    ANG_STD = math.pi
    BOO_STD = 100

    def __init__(self):
        super().__init__()

    def reset(self, initial_state: GameState):
        pass

    def build_obs(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> Any:
        if player.team_num == common_values.ORANGE_TEAM:
            inverted = True
            ball = state.inverted_ball
            # pads = state.inverted_boost_pads
        else:
            inverted = False
            ball = state.ball
            # pads = state.boost_pads

        obs = [
            *ball.position / self.POS_STD,  # 3
            *ball.linear_velocity / self.POS_STD,  # 6
            *ball.angular_velocity / self.ANG_STD  # 9
        ]

        player_car = self._add_player_to_obs(obs, player, ball, inverted)  # 28

        return t(np.array(obs))

    def _add_player_to_obs(self, obs: List, player: PlayerData, ball: PhysicsObject, inverted: bool):
        if inverted:
            player_car = player.inverted_car_data
        else:
            player_car = player.car_data

        obs.extend([
            *player_car.position / self.POS_STD,  # 3
            *player_car.forward(),  # 6
            *player_car.up(),  # 9
            *player_car.linear_velocity / self.POS_STD,  # 12
            *player_car.angular_velocity / self.ANG_STD,  # 15
            player.boost_amount / self.BOO_STD,  # 16
            int(player.on_ground),  # 17
            int(player.has_flip),  # 18
            int(player.is_demoed)  # 19
        ])

        return player_car
