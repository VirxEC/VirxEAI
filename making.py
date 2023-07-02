import struct
from typing import Any, List

import numpy as np
from rlgym_compat import GameState, PlayerData, common_values
from rlgym_sim.utils.math import quat_to_euler
from rlgym_sim.utils.obs_builders import ObsBuilder
from rlgym_sim.utils.reward_functions.combined_reward import CombinedReward
from rlgym_sim.utils.reward_functions.common_rewards import (
    AlignBallGoal, BallYCoordinateReward, FaceBallReward, TouchBallReward,
    VelocityReward)
from rlgym_sim.utils.state_setters import StateSetter, StateWrapper
from rlgym_sim.utils.terminal_conditions.common_conditions import (
    GoalScoredCondition, TimeoutCondition)

REWARD = CombinedReward(
    [
        BallYCoordinateReward(),
        VelocityReward(),
        FaceBallReward(),
        AlignBallGoal(),
        TouchBallReward(1),
    ],
    [
        0.15,
        0.1,
        0.2,
        0.3,
        0.25,
    ]
)

TERMINAL = [
    GoalScoredCondition(),
    TimeoutCondition(225) # 1350 == 1.5 minutes, 4500 == 5 minutes
]


class Obs(ObsBuilder):
    VEL_STD = 2300
    POS_STD = 6000
    CAR_ANG_STD = 5.5
    BALL_ANG_STD = 6
    BOO_STD = 100

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
            self._add_player_to_obs(obs, car, inverted)

        return np.array(obs, dtype=np.float32)

    def _add_player_to_obs(self, obs: List, player: PlayerData, inverted: bool):
        player_car = player.inverted_car_data if inverted else player.car_data

        obs.extend([
            player.team_num,
            *player_car.position / self.POS_STD,
            *player_car.quaternion,
            *player_car.linear_velocity / self.VEL_STD,
            *player_car.angular_velocity / self.CAR_ANG_STD,
            player.boost_amount / self.BOO_STD,
            int(player.on_ground),
            int(player.has_flip),
            int(player.is_demoed),
        ])

BOOL_BYTES = 1
FLOAT_BYTES = 4
VARIANT = 1
UINT_64_BYTES = 8
VEC_3_BYTES = 3 * FLOAT_BYTES
QUAT_BYTES = 4 * FLOAT_BYTES

class RigidBody:
    def __init__(self, sleeping: bool, location: list[float], rotation: list[float], linear_velocity: list[float], angular_velocity: list[float]):
        self.sleeping = sleeping
        self.location = location
        self.rotation = rotation
        self.linear_velocity = linear_velocity
        self.angular_velocity = angular_velocity

    def __repr__(self):
        return f"RigidBody(sleeping={self.sleeping}, location={self.location}, rotation={self.rotation}, linear_velocity={self.linear_velocity}, angular_velocity={self.angular_velocity})"

    def __str__(self):
        return self.__repr__()


class Tick:
    def __init__(self, ball: RigidBody, cars: list[RigidBody]):
        self.ball = ball
        self.cars = cars

    def __repr__(self):
        return f"Tick(ball={self.ball}, cars={self.cars})"

    def __str__(self):
        return self.__repr__()

import multiprocessing

def ball_to_str(ball) -> str:
    return f"PhysicsWrapper(pos={list(ball.position)}, vel={list(ball.linear_velocity)}, ang={list(ball.angular_velocity)})"

def car_to_str(car) -> str:
    return f"CarWrapper(pos={list(car.position)}, rot={list(car.rotation)}, vel={list(car.linear_velocity)}, ang={list(car.angular_velocity)}, boost={car.boost})"

def sw_to_str(sw: StateWrapper) -> str:
    return f"StateWrapper(ball={ball_to_str(sw.ball)}, cars=[{car_to_str(sw.cars[0])}, {car_to_str(sw.cars[1])}])"

class ReplaySetter(StateSetter):
    def __init__(self, replays: list[str], num_replays: int):
        self.replays = replays
        self.num_replays = num_replays

    def reset(self, state_wrapper: StateWrapper):
        while True:
            tick = self._get_random_tick()
            if self._valid_random_tick(tick):
                break

        state_wrapper.ball.position = np.asarray(tick.ball.location)
        state_wrapper.ball.set_lin_vel(*tick.ball.linear_velocity)
        state_wrapper.ball.set_ang_vel(*tick.ball.angular_velocity)

        for i, car in enumerate(state_wrapper.cars):
            car.position = np.asarray(tick.cars[i].location)
            car.set_lin_vel(*tick.cars[i].linear_velocity)
            car.set_ang_vel(*tick.cars[i].angular_velocity)
            car.set_rot(*quat_to_euler(tick.cars[i].rotation))
            car.boost = np.random.randint(0, 100)

        # print(f"\n{multiprocessing.current_process().name}: {tick} | {sw_to_str(state_wrapper)}")

    def _valid_random_tick(self, tick: Tick) -> bool:
        num_cars = len(tick.cars)
        if num_cars == 0 or num_cars % 2 != 0:
            return False

        if tick.ball.sleeping:
            return False

        bad_locations = tuple(round(x) for x in tick.ball.location) == tuple(round(x) for x in tick.cars[0].location) or \
            tuple(round(x) for x in tick.ball.location) == tuple(round(x) for x in tick.cars[1].location) or \
            tuple(round(x) for x in tick.cars[0].location) == tuple(round(x) for x in tick.cars[1].location)
        if bad_locations:
            return False

        return True

    def _get_random_tick(self) -> Tick:
        replay_path = self.replays[np.random.randint(self.num_replays)]
        replay = self._read_replay(replay_path)

        if len(replay) == 0:
            raise Exception(f"replay is empty: {replay_path}")

        replay_len, current_index = self._get_replay_length(replay)
        tick_num = np.random.randint(replay_len)

        for _ in range(tick_num - 1):
            current_index = self._skip_tick(replay, current_index)

        return self._read_tick(replay, current_index)

    @staticmethod
    def _read_replay(replay_path: str) -> bytes:
        with open(replay_path, "rb") as f:
            return f.read()

    @staticmethod
    def _get_replay_length(replay: bytes) -> tuple[int, int]:
        return struct.unpack_from("<Q", replay)[0], UINT_64_BYTES

    @staticmethod
    def _skip_tick(replay: bytes, current_index: int) -> int:
        current_index = ReplaySetter._skip_rigid_body(replay, current_index)
        num_cars = struct.unpack_from("<Q", replay, current_index)[0]
        current_index += UINT_64_BYTES

        for _ in range(num_cars):
            current_index = ReplaySetter._skip_rigid_body(replay, current_index)

        return current_index

    @staticmethod
    def _read_tick(replay: bytes, current_index: int) -> Tick:
        ball, current_index = ReplaySetter._read_rigid_body(replay, current_index)
        num_cars = struct.unpack_from("<Q", replay, current_index)[0]
        current_index += UINT_64_BYTES

        cars = []
        for _ in range(num_cars):
            car, current_index = ReplaySetter._read_rigid_body(replay, current_index)
            cars.append(car)

        return Tick(ball, cars)

    @staticmethod
    def _skip_rigid_body(replay: bytes, current_index: int) -> int:
        current_index += BOOL_BYTES + VEC_3_BYTES + QUAT_BYTES
        lv_variant = struct.unpack_from("<B", replay, current_index)[0]
        current_index += VARIANT

        if lv_variant == 0:
            pass
        elif lv_variant == 1:
            current_index += VEC_3_BYTES
        else:
            raise Exception(f"invalid variant: {lv_variant}")

        av_variant = struct.unpack_from("<B", replay, current_index)[0]
        current_index += VARIANT

        if av_variant == 0:
            pass
        elif av_variant == 1:
            current_index += VEC_3_BYTES
        else:
            raise Exception(f"invalid variant: {av_variant}")

        return current_index

    @staticmethod
    def _read_rigid_body(replay: bytes, current_index: int) -> tuple[RigidBody, int]:
        sleeping = struct.unpack_from("<?", replay, current_index)[0]
        current_index += BOOL_BYTES

        location = struct.unpack_from("<fff", replay, current_index)
        current_index += VEC_3_BYTES

        rotation = struct.unpack_from("<ffff", replay, current_index)
        current_index += QUAT_BYTES

        lv_variant = struct.unpack_from("<B", replay, current_index)[0]
        current_index += VARIANT

        if lv_variant == 0:
            linear_velocity = [0., 0., 0.]
        elif lv_variant == 1:
            linear_velocity = struct.unpack_from("<fff", replay, current_index)
            current_index += VEC_3_BYTES
        else:
            raise Exception(f"invalid variant: {lv_variant}")

        av_variant = struct.unpack_from("<B", replay, current_index)[0]
        current_index += VARIANT

        if av_variant == 0:
            angular_velocity = [0., 0., 0.]
        elif av_variant == 1:
            angular_velocity = struct.unpack_from("<fff", replay, current_index)
            current_index += VEC_3_BYTES
        else:
            raise Exception(f"invalid variant: {av_variant}")

        return RigidBody(sleeping, location, rotation, linear_velocity, angular_velocity), current_index