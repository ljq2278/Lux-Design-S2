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
from ppo.Buffer import Buffer



class PPO_Online_Agent:
    def __init__(self, state_dim, action_dim, env_cfg):
        self.policy = ActorCritic(state_dim, action_dim, env_cfg)

    def update(self, new_params):
        self.policy.load_state_dict(new_params)


class PPO_Offline_Agent:
    def __init__(self, state_dim, action_dim, env_cfg, lr_actor, lr_critic, eps_clip, save_dir='./'):
        self.eps_clip = eps_clip
        self.policy = ActorCritic(state_dim, action_dim, env_cfg)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])
        self.save_dir = save_dir
        self.mseLoss = nn.MSELoss()

    def update_and_get_new_param(self, train_data, K_epochs):
        # Monte Carlo estimate of returns
        # Optimize policy for K epochs
        for _ in range(K_epochs):
            for pid_data in train_data:
                # Evaluating old actions and values
                old_states, old_actions, old_logprobs, old_state_vals, old_rewards, old_done, advantages = \
                    [torch.Tensor(np.array(x)) for x in pid_data]
                logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
                # match state_values tensor dimensions with rewards tensor
                state_values = torch.squeeze(state_values)
                # Finding the ratio (pi_theta / pi_theta__old)
                ratios = torch.exp(logprobs - old_logprobs)
                # Finding Surrogate Loss
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
                # final loss of clipped objective PPO
                loss = -torch.min(surr1, surr2) + 0.5 * self.mseLoss(state_values, old_rewards) - 0.01 * dist_entropy
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
        print('loading unit model .................')
        checkpoint = torch.load(os.path.join(self.save_dir, 'model.pt'))
        self.policy.load_state_dict(checkpoint['policy'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
