import os

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ACTION_NAMES = ["throttle", "steer", "pitch", "yaw", "roll", "jump", "boost", "handbrake"]
GAMMA = 0.98
EPS = 0.2
MAX_GRAD_NORM = 0.5
STATE_DIM = 28
N_ACTIONS = 8


class Mish(nn.Module):
    def forward(self, input_): return input_ * torch.tanh(F.softplus(input_))


# helper function to convert numpy arrays to tensors
def t(x): return torch.from_numpy(x).float()


def Actor(state_dim, n_actions, activation=nn.Tanh):
    layers = [
        nn.Linear(state_dim, state_dim),
        activation()
    ]

    for _ in range(n_actions):
        layers.extend([
            nn.Linear(state_dim, state_dim),
            activation()
        ])

    layers.extend([
        nn.Linear(state_dim, n_actions),
    ])

    return nn.Sequential(*layers)


def Critic(state_dim, activation=nn.Tanh):
    layers = [
        nn.Linear(state_dim, state_dim),
        activation()
    ]

    # for _ in range(round(state_dim / 2)):
    layers.extend([
        nn.Linear(state_dim, state_dim),
        activation()
    ])

    layers.extend([
        nn.Linear(state_dim, 1)
    ])

    return nn.Sequential(*layers)


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


torch.autograd.set_detect_anomaly(True)


class Player:
    def __init__(self, new_ai, base_folder, train=False):
        print(f"Bulding player...")
        self.base_folder = base_folder
        self.total_reward = [0, 0]

        self.actors = []
        self.critics = []
        self.adam_actors = []
        self.adam_critics = []

        self.dists = []
        self.actions = []
        self.probs = []
        self.prob_acts = []
        self.prev_prob_acts = []

        for i in range(N_ACTIONS):
            if new_ai:
                self.actors.append(Actor(STATE_DIM, 2, activation=Mish))
                self.critics.append(Critic(STATE_DIM, activation=Mish))
            else:
                action_name = ACTION_NAMES[i]
                self.actors.append(nn.Sequential().load_state_dict(torch.load(os.path.join(self.base_folder, "models", f"actor_{action_name}.pt"))))
                self.actors[i].train(train)
                self.critics.append(nn.Sequential().load_state_dict(torch.load(os.path.join(self.base_folder, "models", f"critic_{action_name}.pt"))))
                self.critics[i].train(train)

            self.adam_actors.append(torch.optim.Adam(self.actors[i].parameters(), lr=3e-4))  # learning rate: 3e-4
            self.adam_critics.append(torch.optim.Adam(self.critics[i].parameters(), lr=1e-3))  # learning rate: 1e-3
        
        print("Actor state dict:")
        for param_tensor in self.actors[0].state_dict():
            print(param_tensor, "\t", self.actors[0].state_dict()[param_tensor].size())

        print("Critic state dict:")
        for param_tensor in self.critics[0].state_dict():
            print(param_tensor, "\t", self.critics[0].state_dict()[param_tensor].size())

    def step(self, num_players, state):
        self.dists = []
        self.actions = []
        self.probs = []
        self.prob_acts = []
        actions = []

        for num_player in range(num_players):
            self.dists.append([])
            self.actions.append([])
            self.probs.append([])
            self.prob_acts.append([])
            actions.append([])

            for i in range(N_ACTIONS):
                self.probs[num_player].append(self.actors[i](state[num_player]))
                probs = torch.clamp(self.probs[num_player][i].clone(), -1, 1) / 2 + 1
                self.dists[num_player].append(torch.distributions.Categorical(probs=probs))
                self.actions[num_player].append(self.dists[num_player][i].sample())
                self.prob_acts[num_player].append(self.dists[num_player][i].log_prob(self.actions[num_player][i]))
                actions[num_player].append(self.actions[num_player][i].detach().data.numpy())

        return actions

    def learn(self, num_players, state, last_state, reward, w, s):
        for num_player in range(num_players):
            for i in range(N_ACTIONS):
                name = ACTION_NAMES[i]
                advantage = reward[num_player] + GAMMA*self.critics[i](state[num_player]) - self.critics[i](last_state[num_player])

                w[num_player].add_scalar(f"loss/{name}_advantage", advantage, global_step=s)
                w[num_player].add_scalar(f"actions/{name}_prob", self.dists[num_player][i].probs[1], global_step=s)

                if len(self.prev_prob_acts) > 0:
                    actor_loss = policy_loss(self.prev_prob_acts[num_player][i].detach(), self.prob_acts[num_player][i], advantage.detach(), EPS)
                    w[num_player].add_scalar(f"loss/{name}_actor_loss", actor_loss, global_step=s)
                    self.adam_actors[i].zero_grad()
                    actor_loss.backward()
                    clip_grad_norm_(self.adam_actors[i], MAX_GRAD_NORM)
                    w[num_player].add_histogram(f"gradients/{name}_actor", torch.cat([p.grad.view(-1) for p in self.actors[i].parameters()]), global_step=s)
                    self.adam_actors[i].step()

                    critic_loss = advantage.pow(2).mean()
                    w[num_player].add_scalar(f"loss/{name}_critic_loss", critic_loss, global_step=s)
                    self.adam_critics[i].zero_grad()
                    critic_loss.backward()
                    clip_grad_norm_(self.adam_critics[i], MAX_GRAD_NORM)
                    w[num_player].add_histogram(f"gradients/{name}_critic", torch.cat([p.data.view(-1) for p in self.critics[i].parameters()]), global_step=s)
                    self.adam_critics[i].step()

        for num_player in range(num_players):
            self.total_reward[num_player] += reward[num_player]

        self.prev_prob_acts = [[prob_act.clone() for prob_act in prob_acts] for prob_acts in self.prob_acts]

    def end_episode(self, num_players, w, s):
        for num_player in range(num_players):
            w[num_player].add_scalar("reward/episode_reward", self.total_reward[num_player], global_step=s)

        for i in range(N_ACTIONS):
            action_name = ACTION_NAMES[i]
            torch.save(self.actors[i].state_dict(), os.path.join(self.base_folder, "models", f"actor_{action_name}.pt"))
            torch.save(self.critics[i].state_dict(), os.path.join(self.base_folder, "models", f"critic_{action_name}.pt"))
        self.prev_prob_acts = []
