import math
from typing import Any, List

import numpy as np
from rlgym_compat import common_values
from rlgym_compat.physics_object import PhysicsObject
from rlgym_compat.player_data import PlayerData
from rlgym_compat.game_state import GameState


def scalar_projection(vec, dest_vec):
    norm = np.linalg.norm(dest_vec)
    return 0 if norm == 0 else np.dot(vec, dest_vec) / norm

class Reward():
    def __init__(self):
        super().__init__()
        self.last_ball_touch = False

    def reset(self, initial_state: GameState, optional_data=None):
        self.last_ball_touch = False

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray, optional_data=None):
        # Vector version of v=d/t <=> t=d/v <=> 1/t=v/d
        # Max value should be max_speed / ball_radius = 2300 / 94.75 = 24.3 / 30.375 = 0.8
        # Used to guide the agent towards the ball
        inv_t = scalar_projection(player.car_data.linear_velocity, state.ball.position - player.car_data.position) / 30.375

        # If an action caused the bot to touch the ball, give the bot a bonus reward of 0.2
        ball_touch = 0.2 if player.ball_touched and not self.last_ball_touch else 0
        self.last_ball_touch = player.ball_touched

        return inv_t + ball_touch

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray, optional_data=None):
        return 0


class Obs():
    POS_STD = 2300
    ANG_STD = math.pi
    BOO_STD = 100

    def __init__(self):
        super().__init__()

    def reset(self, initial_state: GameState):
        pass

    def build_obs(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> Any:
        inverted = player.team_num == common_values.ORANGE_TEAM
        ball = state.inverted_ball if inverted else state.ball
        # pads = state.inverted_boost_pads if inverted else state.boost_pads

        obs = [
            *ball.position / self.POS_STD,  # 3
            *ball.linear_velocity / self.POS_STD,  # 6
            *ball.angular_velocity / self.ANG_STD  # 9
        ]

        self._add_player_to_obs(obs, player, ball, inverted)  # 28

        return np.array(obs)

    def _add_player_to_obs(self, obs: List, player: PlayerData, ball: PhysicsObject, inverted: bool):
        player_car = player.inverted_car_data if inverted else player.car_data

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
