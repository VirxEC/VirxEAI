from typing import Any, List

import numpy as np
from rlgym_compat import GameState, PhysicsObject, PlayerData, common_values
from rlgym_sim.utils.obs_builders import ObsBuilder
from rlgym_sim.utils.reward_functions.combined_reward import CombinedReward
from rlgym_sim.utils.reward_functions.common_rewards import (
    AlignBallGoal, BallYCoordinateReward, FaceBallReward, TouchBallReward)

REWARD = CombinedReward(
    [
        BallYCoordinateReward(),
        FaceBallReward(),
        AlignBallGoal(),
        TouchBallReward(),
    ],
    [
        0.1,
        0.15,
        0.5,
        0.25,
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
        pads = state.inverted_boost_pads if inverted else state.boost_pads

        obs = [
            *pads,
            *previous_action,
            *ball.position / self.POS_STD,
            *ball.linear_velocity / self.POS_STD,
            *ball.angular_velocity / self.BALL_ANG_STD,
        ]

        for car in state.players:
            self._add_player_to_obs(obs, car, ball, inverted)

        return np.array(obs)

    def _add_player_to_obs(self, obs: List, player: PlayerData, ball: PhysicsObject, inverted: bool):
        player_car = player.inverted_car_data if inverted else player.car_data

        obs.extend([
            *player_car.position / self.POS_STD,
            *player_car.quaternion,
            *player_car.linear_velocity / self.VEL_STD,
            *player_car.angular_velocity / self.CAR_ANG_STD,
            player.boost_amount / self.BOO_STD,
            int(player.on_ground),
            int(player.has_flip),
            int(player.is_demoed),
        ])
