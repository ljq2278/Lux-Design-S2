from copy import deepcopy
from typing import List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.optim import Adam
from torch.distributions import Categorical

class UnitAgent:
    """Agent that can interact with environment from pettingzoo"""

    def __init__(self, obs_dim, act_dim, actor_lr, critic_lr):
        self.actor = MLPNetwork(obs_dim, act_dim)
        self.critic = CriMLPNetwork(obs_dim, 1)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=critic_lr)


    def update_actor(self, loss):
        self.actor_optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()

    def update_critic(self, loss):
        self.critic_optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()


class MLPNetwork(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=124, non_linear=nn.ReLU()):
        super(MLPNetwork, self).__init__()

        self.net = nn.Sequential(
            # nn.Linear(in_dim, hidden_dim),
            # non_linear,
            # nn.Linear(hidden_dim, hidden_dim),
            # non_linear,
            nn.Linear(in_dim, out_dim),
            nn.Softmax(),
        ).apply(self.init)

    @staticmethod
    def init(m):
        """init parameter of the module"""
        gain = nn.init.calculate_gain('relu')
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=gain)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        dist = Categorical(self.net(x))
        return dist


class CriMLPNetwork(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=80, non_linear=nn.ReLU()):
        super(CriMLPNetwork, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            non_linear,
            # nn.Linear(hidden_dim, hidden_dim),
            # non_linear,
            nn.Linear(hidden_dim, out_dim),
        ).apply(self.init)

    @staticmethod
    def init(m):
        """init parameter of the module"""
        gain = nn.init.calculate_gain('relu')
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=gain)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        return self.net(x)
