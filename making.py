from typing import Any, List

import numpy as np
from rlgym_compat import common_values
from rlgym_compat.game_state import GameState
from rlgym_compat.physics_object import PhysicsObject
from rlgym_compat.player_data import PlayerData
from rlgym_sim.utils.obs_builders import ObsBuilder
from rlgym_sim.utils.reward_functions.combined_reward import CombinedReward
from rlgym_sim.utils.reward_functions.common_rewards.player_ball_rewards import (
    TouchBallReward, VelocityPlayerToBallReward)

REWARD = CombinedReward(
    [
        VelocityPlayerToBallReward(),
        TouchBallReward(),
    ],
    [
        0.8,
        0.2,
    ]
)


class Obs(ObsBuilder):
    VEL_STD = 2300
    POS_STD = 6000
    CAR_ANG_STD = 5.5
    BALL_ANG_STD = 6
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
            *ball.angular_velocity / self.BALL_ANG_STD  # 9
        ]

        self._add_player_to_obs(obs, player, ball, inverted)  # 28

        return np.array(obs)

    def _add_player_to_obs(self, obs: List, player: PlayerData, ball: PhysicsObject, inverted: bool):
        player_car = player.inverted_car_data if inverted else player.car_data

        obs.extend([
            *player_car.position / self.POS_STD,  # 3
            *player_car.forward(),  # 6
            *player_car.up(),  # 9
            *player_car.linear_velocity / self.VEL_STD,  # 12
            *player_car.angular_velocity / self.CAR_ANG_STD,  # 15
            player.boost_amount / self.BOO_STD,  # 16
            int(player.on_ground),  # 17
            int(player.has_flip),  # 18
            int(player.is_demoed)  # 19
        ])
