import math
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


def train_model(s, name, actor, critic, adam_actor, adam_critic, states, actions, rewards):
    import torch

    last_prob_act = torch.distributions.Categorical(probs=torch.clamp(actor(states[0]), -1, 1) / 2 + 1).log_prob(actions[0])

    for j in range(1, len(states)):
        try:
            last_state = states[j-1]
            state = states[j]
            action = actions[j]
            reward = rewards[j]
        except IndexError:
            continue

        advantage = reward + GAMMA*critic(state) - critic(last_state)

        probs = torch.clamp(actor(state), -1, 1) / 2 + 1
        dist = torch.distributions.Categorical(probs=probs)
        prob_act = dist.log_prob(action)

        actor_loss = policy_loss(last_prob_act.detach(), prob_act, advantage.detach(), EPS)
        adam_actor.zero_grad()
        actor_loss.backward()
        clip_grad_norm_(adam_actor, MAX_GRAD_NORM)
        adam_actor.step()

        critic_loss = advantage.pow(2).mean()
        adam_critic.zero_grad()
        critic_loss.backward()
        clip_grad_norm_(adam_critic, MAX_GRAD_NORM)
        adam_critic.step()

        last_prob_act = prob_act

        # w[num_player].add_scalar(f"loss/{name}_advantage", advantage, global_step=s)
        # w[num_player].add_scalar(f"actions/{name}_prob", dist.probs[1], global_step=s)
        # w[num_player].add_scalar(f"loss/{name}_actor_loss", actor_loss, global_step=s)
        # w[num_player].add_histogram(f"gradients/{name}_actor", torch.cat([p.grad.view(-1) for p in self.actors[i].parameters()]), global_step=s)
        # w[num_player].add_scalar(f"loss/{name}_critic_loss", critic_loss, global_step=s)
        # w[num_player].add_histogram(f"gradients/{name}_critic", torch.cat([p.data.view(-1) for p in self.critics[i].parameters()]), global_step=s)


class Player:
    def __init__(self, num_players, new_ai, base_folder, train=False):
        print(f"Bulding player...")
        self.cpu_count = max(os.cpu_count() - 4, 1)
        torch.set_num_threads(1)
        self.base_folder = base_folder
        self.num_players = num_players

        self.actors = []
        self.critics = []
        self.adam_actors = []
        self.adam_critics = []
        self.prepare_for_new_episode()

        for i in range(N_ACTIONS):
            if new_ai:
                self.actors.append(Actor(STATE_DIM, activation=Mish))
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

    def prepare_for_new_episode(self):
        self.states = [[] for _ in range(self.num_players)]
        self.actions = [[[] for _ in range(N_ACTIONS)] for _ in range(self.num_players)]
        self.rewards = [[] for _ in range(self.num_players)]

    def step(self, states):
        actions = []

        for num_player in range(self.num_players):
            state = states[num_player]

            self.states[num_player].append(state)

            actions.append([])

            for i in range(N_ACTIONS):
                probs = torch.clamp(self.actors[i](state), -1, 1) / 2 + 1
                dist = torch.distributions.Categorical(probs=probs)

                action = dist.sample()
                self.actions[num_player][i].append(action)
                actions[num_player].append(action.detach().data.numpy())

        return actions

    def add_reward(self, rewards):
        for num_player in range(self.num_players):
            self.rewards[num_player].append(rewards[num_player])

    def get_total_rewards(self):
        return [sum(player_rewards) for player_rewards in self.rewards]

    def learn(self, w, s):
        for num_player in range(self.num_players):
            print(f"Training player {num_player}...")
            for i in range(N_ACTIONS):
                name = ACTION_NAMES[i]
                print(f"Training {name}...")

                num_states = len(self.states[num_player])
                last_prob_act = torch.distributions.Categorical(probs=torch.clamp(self.actors[i](self.states[num_player][0]), -1, 1) / 2 + 1).log_prob(self.actions[num_player][0][i])
                
                for j in range(1, num_states):
                    try:
                        last_state = self.states[num_player][j-1]
                        state = self.states[num_player][j]
                        action = self.actions[num_player][j][i]
                        reward = self.rewards[num_player][j]
                    except IndexError:
                        continue

                    advantage = reward + GAMMA*self.critics[i](state) - self.critics[i](last_state)

                    probs = torch.clamp(self.actors[i](state), -1, 1) / 2 + 1
                    dist = torch.distributions.Categorical(probs=probs)
                    prob_act = dist.log_prob(action)

                    actor_loss = policy_loss(last_prob_act.detach(), prob_act, advantage.detach(), EPS)
                    self.adam_actors[i].zero_grad()
                    actor_loss.backward()
                    clip_grad_norm_(self.adam_actors[i], MAX_GRAD_NORM)
                    self.adam_actors[i].step()

                    critic_loss = advantage.pow(2).mean()
                    self.adam_critics[i].zero_grad()
                    critic_loss.backward()
                    clip_grad_norm_(self.adam_critics[i], MAX_GRAD_NORM)
                    self.adam_critics[i].step()

                    last_prob_act = prob_act

    def end_episode(self, w, s):
        total_rewards = self.get_total_rewards()
        for num_player in range(self.num_players):
            w[num_player].add_scalar("reward/episode_reward", total_rewards[num_player], global_step=s)

        models = os.path.join(self.base_folder, "models")
        if not os.path.isdir(models):
            os.mkdir(models)

        for i in range(N_ACTIONS):
            action_name = ACTION_NAMES[i]
            torch.save(self.actors[i].state_dict(), os.path.join(models, f"actor_{action_name}.pt"))
            torch.save(self.critics[i].state_dict(), os.path.join(models, f"critic_{action_name}.pt"))

        self.prepare_for_new_episode()
