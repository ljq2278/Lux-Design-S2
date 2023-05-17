import os
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.optim import Adam
from torch.distributions import Categorical
import numpy as np
from wrappers.obs_space_conv import ObsSpace
# from wrappers.act_space_levels import ActSpaceFactoryDemand
from actcrt_model.ac_model_conv import ActorCritic
from ppo_conv.Buffer import Buffer
import copy


class CentralOnlineAgent:
    def __init__(self, state_dim, f_action_dim, u_action_dim, env_cfg):
        self.policy = ActorCritic(state_dim, f_action_dim, u_action_dim, env_cfg)

    def update(self, new_params):
        self.policy.load_state_dict(new_params)


class CentralOfflineAgent:
    def __init__(self, state_dim, f_action_dim, u_action_dim, env_cfg, lr_actor, lr_critic, eps_clip, save_dir='./'):
        self.eps_clip = eps_clip
        self.obs_space = ObsSpace(env_cfg)
        self.policy = ActorCritic(state_dim, f_action_dim, u_action_dim, env_cfg)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])
        self.save_dir = save_dir
        self.mseLoss = nn.MSELoss()

    def update_and_get_new_param(self, train_data, K_epochs):
        # Monte Carlo estimate of returns
        # Optimize policy for K epochs
        for epochs_i in range(K_epochs):
            print('train_epochs: ', epochs_i)
            for pid_data in train_data:
                # Evaluating old actions and values
                old_states, old_state_vals, old_f_actions, old_f_logprobs, old_u_actions, old_u_logprobs, old_rewards, old_done, advantages = [torch.Tensor(np.array(x)) for x in pid_data]
                old_f_masks, old_u_masks = old_states[:, self.obs_space.f_pos_dim_start, :, :], old_states[:, self.obs_space.u_pos_dim_start, :, :]
                state_values, f_logprobs, f_dist_entropy, u_logprobs, u_dist_entropy = self.policy.evaluate(old_states, old_f_actions, old_u_actions)
                # match state_values tensor dimensions with rewards tensor
                state_values = torch.squeeze(state_values)
                # Finding the ratio (pi_theta / pi_theta__old)
                f_ratios, u_ratios = torch.exp(f_logprobs - old_f_logprobs), torch.exp(u_logprobs - old_u_logprobs)
                # Finding Surrogate Loss
                us_advantages = advantages.unsqueeze(dim=1).unsqueeze(dim=1)
                f_surr1, u_surr1 = f_ratios * us_advantages * old_f_masks, u_ratios * us_advantages * old_u_masks
                f_surr2, u_surr2 = torch.clamp(f_ratios, 1 - self.eps_clip, 1 + self.eps_clip) * us_advantages, \
                                   torch.clamp(u_ratios, 1 - self.eps_clip, 1 + self.eps_clip) * us_advantages
                # final loss of clipped objective PPO
                f_loss = -torch.min(f_surr1, f_surr2) + 0.5 * self.mseLoss(state_values, old_rewards) - 0.01 * f_dist_entropy
                u_loss = -torch.min(u_surr1, u_surr2) + 0.5 * self.mseLoss(state_values, old_rewards) - 0.01 * u_dist_entropy
                loss = f_loss + u_loss
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
        }, os.path.join(self.save_dir, 'f_model.pt'))

    def load(self):
        """init maddpg using the model saved in `file`"""
        print('loading factory model .................')
        checkpoint = torch.load(os.path.join(self.save_dir, 'f_model.pt'))
        self.policy.load_state_dict(checkpoint['policy'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
