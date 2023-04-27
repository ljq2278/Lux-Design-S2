from copy import deepcopy
from typing import List
import os
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.optim import Adam
from torch.distributions import Categorical
import numpy as np

from actcrt_model.ac_model import ActorCritic
from ppo.UnitBuffer import Buffer


class A3C_Online_Agent:
    def __init__(self, state_dim, action_dim):
        self.policy = ActorCritic(state_dim, action_dim)

    def update(self, new_params):
        self.policy.load_state_dict(new_params)


class A3C_Offline_Agent:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, eps_clip, save_dir='./'):
        self.eps_clip = eps_clip
        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])
        self.save_dir = save_dir
        self.mseLoss = nn.MSELoss()

    def update_and_get_new_param(self, train_data):
        # Monte Carlo estimate of returns
        # Optimize policy for K epochs
        old_states, old_actions, old_logprobs, old_state_vals, old_rewards, old_done, advantages = [], [], [], [], [], [], []
        for pid_data in train_data:
            old_states += pid_data[0]
            old_actions += pid_data[1]
            old_logprobs += pid_data[2]
            old_state_vals += pid_data[3]
            old_rewards += pid_data[4]
            old_done += pid_data[5]
            advantages += pid_data[6]
        old_states, old_actions, old_logprobs, old_state_vals, old_rewards, old_done, advantages = \
            torch.Tensor(np.array(old_states)), torch.Tensor(np.array(old_actions)), \
            torch.Tensor(np.array(old_logprobs)), \
            torch.Tensor(np.array(old_state_vals)), torch.Tensor(np.array(old_rewards)), \
            torch.Tensor(np.array(old_done)), torch.Tensor(np.array(advantages))
        logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
        # match state_values tensor dimensions with rewards tensor
        state_values = torch.squeeze(state_values)
        # Finding the ratio (pi_theta / pi_theta__old)
        # actor_loss = torch.log(-logprobs) * advantages
        actor_loss = -logprobs * advantages
        critic_loss = 0.5 * self.mseLoss(state_values, old_rewards)
        # final loss of clipped objective PPO
        loss = actor_loss + critic_loss
        # take gradient step
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

        return self.policy.state_dict()

    def save(self):
        """save actor parameters of all agents and training reward to `res_dir`"""
        torch.save({
            'policy': self.policy.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, os.path.join(self.save_dir, 'model.pt'))

    def load(self):
        """init maddpg using the model saved in `file`"""
        print('loading model .................')
        checkpoint = torch.load(os.path.join(self.save_dir, 'model.pt'))
        self.policy.load_state_dict(checkpoint['policy'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
