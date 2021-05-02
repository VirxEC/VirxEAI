import math
import os

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from MPFramework import MPFProcess, MPFProcessHandler

ACTION_NAMES = ["throttle", "steer", "pitch", "yaw", "roll", "jump", "boost", "handbrake"]
GAMMA = 0.98
EPS = 0.2
MAX_GRAD_NORM = 0.5
STATE_DIM = 28
N_ACTIONS = 8
torch.set_num_threads(2)


class Mish(nn.Module):
    def forward(self, input_): return input_ * torch.tanh(F.softplus(input_))


# helper function to convert numpy arrays to tensors
def t(x): return torch.from_numpy(x).float()


def Actor(state_dim, activation=nn.Tanh):
    return nn.Sequential(
        nn.Linear(state_dim, round(state_dim / 2)),
        activation(),
        nn.Linear(round(state_dim / 2), round(state_dim / 2)),
        nn.Dropout(0.1),
        activation(),
        nn.Linear(round(state_dim / 2), round(state_dim / 4)),
        nn.Dropout(0.2),
        activation(),
        nn.Linear(round(state_dim / 4), 2),
    )


def Critic(state_dim, activation=nn.Tanh):
    return nn.Sequential(
        nn.Linear(state_dim, round(state_dim / 2)),
        activation(),
        nn.Linear(round(state_dim / 2), round(state_dim / 4)),
        nn.Dropout(0.2),
        activation(),
        nn.Linear(round(state_dim / 4), 1)
    )


def cap(x, low, high):
    return ((x if x < high else high) if x > low else low)


def clip_grad_norm_(module, max_grad_norm):
    nn.utils.clip_grad_norm_([p for g in module.param_groups for p in g["params"]], max_grad_norm)


def policy_loss(old_log_prob, log_prob, advantage, eps):
    ratio = (log_prob - old_log_prob).exp()
    clipped = torch.clamp(ratio, 1-eps, 1+eps)*advantage

    m = torch.min(ratio*advantage, clipped)
    return -m


CONTROLLER_RANGE = [-1, 1]
def get_range(x):
    return CONTROLLER_RANGE[x]


def numpy_to_controller(actions):
    return [
        get_range(actions[0]),  # throttle
        get_range(actions[1]),  # steer
        get_range(actions[2]),  # pitch
        get_range(actions[3]),  # yaw
        get_range(actions[4]),  # roll
        bool(actions[5]),       # jump
        bool(actions[6]),       # boost
        bool(actions[7])        # handbrake
    ]

class ModelProcesser(MPFProcess):
    def __init__(self, name, num_players, new_ai, models_folder, train):
        #We set loop wait period to 1 here to pretend the process is doing something intensive.
        super().__init__(process_name=f"{name}_handler")
        self.name = name
        self.num_players = num_players
        self.models_folder = models_folder
        self.prepare_for_new_episode()

        if new_ai:
            self.actor = Actor(STATE_DIM, activation=Mish)
            self.critic = Critic(STATE_DIM, activation=Mish)
        else:
            self.actor = nn.Sequential().load_state_dict(torch.load(os.path.join(self.models_folder, f"actor_{self.name}.pt")))
            self.critic = nn.Sequential().load_state_dict(torch.load(os.path.join(self.models_folder, f"critic_{self.name}.pt")))

        self.actor.train(train)
        self.critic.train(train)

        self.adam_actor = torch.optim.Adam(self.actor.parameters(), lr=3e-4)  # learning rate: 3e-4
        self.adam_critic = torch.optim.Adam(self.critic.parameters(), lr=1e-3)  # learning rate: 1e-3

    def run(self):
        """
        The function to be called when a process is started.
        :return: None
        """

        try:
            #We import everything important here to ensure that the libraries we need will be imported into the new
            #process memory instead of the main process memory.
            import logging
            import sys
            # import time
            import traceback

            from MPFramework import MPFResultPublisher, MPFTaskChecker

            #This are our i/o objects for interfacing with the main process.
            self.task_checker = MPFTaskChecker(self._inp, self.name)
            self.results_publisher = MPFResultPublisher(self._out, self.name)

            self._MPFLog = logging.getLogger("MPFLogger")
            self._MPFLog.debug("MPFProcess initializing...")

            #Initialize.
            self._MPFLog.debug("MPFProcess {} has successfully initialized".format(self.name))

            while True:
                #Check for new inputs from the main process.
                data_packet = self.task_checker._input_queue.get()
                header, data = data_packet()
                self.task_checker._update_data(data)
                self.task_checker.header = header
                self.task_checker._check_for_terminal_message(header, data)
                data_packet.cleanup()
                del data_packet

                self._MPFLog.debug("Process {} got update {}".format(self.name, self.task_checker.header))

                #If we are told to stop running, do so.
                if self.task_checker.header == MPFProcess.STOP_KEYWORD:
                    self._MPFLog.debug("PROCESS {} RECEIVED STOP SIGNAL!".format(self.name))
                    self._successful_termination = True
                    raise sys.exit(0)

                #Otherwise, update with the latest main process message.
                self._MPFLog.debug("Process {} sending update to subclass".format(self.name))
                self.update(self.task_checker.header, self.task_checker.latest_data)

        except:
            #Catch-all because I'm lazy.
            error = traceback.format_exc()
            if not self._successful_termination:
                self._MPFLog.critical("MPFPROCESS {} HAS CRASHED!\n"
                               "EXCEPTION TRACEBACK:\n"
                               "{}".format(self.name, error))

        finally:
            #Clean everything up and terminate.
            if self.task_checker is not None:
                self._MPFLog.debug("MPFProcess {} Cleaning task checker...".format(self.name))
                self.task_checker.cleanup()
                del self.task_checker
                self._MPFLog.debug("MPFProcess {} has cleaned its task checker!".format(self.name))

            if self.results_publisher is not None:
                del self.results_publisher

            self._MPFLog.debug("MPFProcess {} Cleaning up...".format(self.name))
            self._MPFLog.debug("MPFProcess {} Exiting!".format(self.name))
            return
    
    def update(self, header, data):
        if 'new_episode' in header:
            self.prepare_for_new_episode()
        elif 'step' in header:
            self.step(data)
        elif 'add_rewards' in header:
            self.add_rewards(self, rewards)
        elif 'learn' in header:
            self.learn()
        elif 'end_episode' in header:
            self.end_episode()
        
    def prepare_for_new_episode(self):
        self.states = [[] for _ in range(self.num_players)]
        self.actions = [[] for _ in range(self.num_players)]
        self.rewards = [[] for _ in range(self.num_players)]

    def step(self, states):
        for num_player in range(self.num_players):
            state = t(states[num_player])
            self.states[num_player].append(state)

            probs = torch.clamp(self.actor(state), -1, 1) / 2 + 1
            dist = torch.distributions.Categorical(probs=probs)

            action = dist.sample()
            self.actions[num_player].append(action)
            self.results_publisher.publish(action.detach().numpy(), header=num_player)
    
    def add_rewards(self, rewards):
        for num_player in range(self.num_players):
            self.rewards[num_player].append(rewards[num_player])

    def learn(self):
        for num_player in range(self.num_players):
            last_prob_act = torch.distributions.Categorical(probs=torch.clamp(self.actor(self.states[num_player][0]), -1, 1) / 2 + 1).log_prob(self.actions[num_player][0])

            for j in range(1, len(self.states[num_player])):
                try:
                    last_state = self.states[num_player][j-1]
                    state = self.states[num_player][j]
                    action = self.actions[num_player][j]
                    reward = self.rewards[num_player][j]
                except IndexError:
                    continue

                advantage = reward + GAMMA*self.critic(state) - self.critic(last_state)

                probs = torch.clamp(self.actor(state), -1, 1) / 2 + 1
                dist = torch.distributions.Categorical(probs=probs)
                prob_act = dist.log_prob(action)

                actor_loss = policy_loss(last_prob_act.detach(), prob_act, advantage.detach(), EPS)
                self.adam_actor.zero_grad()
                actor_loss.backward()
                clip_grad_norm_(self.adam_actor, MAX_GRAD_NORM)
                self.adam_actor.step()

                critic_loss = advantage.pow(2).mean()
                self.adam_critic.zero_grad()
                critic_loss.backward()
                clip_grad_norm_(self.adam_critic, MAX_GRAD_NORM)
                self.adam_critic.step()

                last_prob_act = prob_act

        self.results_publisher.publish(1, header="done")

    def end_episode(self):
        torch.save(self.actor.state_dict(), os.path.join(self.models_folder, f"actor_{self.name}.pt"))
        torch.save(self.critic.state_dict(), os.path.join(self.models_folder, f"critic_{self.name}.pt"))
        self.prepare_for_new_episode()

        self.results_publisher.publish(1, header="done")

if __name__ != "__mp_main__":
    class Player:
        def __init__(self, num_players, new_ai, base_folder, train=False):
            print(f"Bulding player...")

            self.models_folder = os.path.join(base_folder, "models")
            if not os.path.isdir(self.models_folder):
                os.mkdir(self.models_folder)

            self.num_players = num_players
            self.action_processers = []
            self.prepare_for_new_episode()

            for i in range(N_ACTIONS):
                action_name = ACTION_NAMES[i]
                p = MPFProcessHandler()
                p.setup_process(ModelProcesser(action_name, self.num_players, new_ai, self.models_folder, train))
                self.action_processers.append(p)

        def prepare_for_new_episode(self):
            self.rewards = [[] for _ in range(self.num_players)]

        def step(self, states):
            actions = [[] for _ in range(self.num_players)]

            for i in range(N_ACTIONS):
                self.action_processers[i].put("step", states)

            for i in range(N_ACTIONS):
                for num_player in range(self.num_players):
                    action = self.action_processers[i]._output_queue.get()()
                    actions[action[0]].append(action[1])

            return actions

        def add_reward(self, rewards):
            for num_player in range(self.num_players):
                self.rewards[num_player].append(rewards[num_player])

            for i in range(N_ACTIONS):
                self.action_processers[i].put("add_reward", rewards)

        def get_total_rewards(self):
            return [sum(player_rewards) for player_rewards in self.rewards]

        def learn(self, w, s):
            for i in range(N_ACTIONS):
                self.action_processers[i].put("learn", 1)
                
            for i in range(N_ACTIONS):
                self.action_processers[i]._output_queue.get()

        def end_episode(self, w, s):
            total_rewards = self.get_total_rewards()
            for num_player in range(self.num_players):
                w[num_player].add_scalar("reward/episode_reward", total_rewards[num_player], global_step=s)

            for i in range(N_ACTIONS):
                self.action_processers[i].put("end_episode", 1)

            self.prepare_for_new_episode()
                
            for i in range(N_ACTIONS):
                self.action_processers[i]._output_queue.get()
                name = ACTION_NAMES[i]

        def close(self):
            for process in self.action_processers:
                process.close()
