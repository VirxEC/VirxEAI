import numpy as np
from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket
from rlgym_compat import GameState

from agent import Agent
from making import Obs, Reward


class RLGymExampleBot(BaseAgent):
    def __init__(self, name, team, index):
        super().__init__(name, team, index)

        self.obs_builder = Obs()
        # Your neural network logic goes inside the Agent class, go take a look inside src/agent.py
        self.agent = Agent()
        # Adjust the tickskip if your agent was trained with a different value
        self.tick_skip = 8

        self.game_state: GameState = None
        self.controls = None
        self.prev_action = None
        self.ticks = 0
        self.prev_time = 0
        print('RLGymExampleBot Ready - Index:', index)

    def initialize_agent(self):
        # Initialize the rlgym GameState object now that the game is active and the info is available
        self.game_state = GameState(self.get_field_info())
        self.ticks = self.tick_skip  # So we take an action the first tick
        self.prev_time = 0
        self.controls = SimpleControllerState()
        self.prev_action = np.array([0, 0, 0, 0, 0, 0, 0, 0])

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        cur_time = packet.game_info.seconds_elapsed
        delta = cur_time - self.prev_time
        self.prev_time = cur_time

        self.ticks += delta // 0.008  # Smaller than 1/120 on purpose

        if self.ticks >= self.tick_skip:
            self.ticks = 0

            self.game_state.decode(packet)

            player = self.game_state.players[self.index]
            opponents = [p for p in self.game_state.players if p.team_num != self.team]

            # Another option is to focus on the opponent closest to the ball
            # you can use any logic you see fit to choose the op you want to focus on
            closest_op = min(opponents, key=lambda p: np.linalg.norm(self.game_state.ball.position - p.car_data.position))
            # self.renderer.draw_string_3d(closest_op.car_data.position, 2, 2, "CLOSEST", self.renderer.white())

            self.game_state.players = [player, closest_op]

            obs = self.obs_builder.build_obs(player, self.game_state, self.prev_action)
            action = self.agent.act(obs)
            self.update_controls(action)

        return self.controls

    def update_controls(self, action):
        self.prev_action = action

        self.controls.throttle = action[0]
        self.controls.steer = action[1]
        self.controls.pitch = action[2]
        self.controls.yaw = action[3]
        self.controls.roll = action[4]
        self.controls.jump = action[5] > 0
        self.controls.boost = action[6] > 0
        self.controls.handbrake = action[7] > 0
