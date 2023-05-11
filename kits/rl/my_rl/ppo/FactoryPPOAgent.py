import os
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.optim import Adam
from torch.distributions import Categorical
import numpy as np
from wrappers.obs_space_levels import ObsSpaceFactory
from wrappers.act_space_levels import ActSpaceFactoryDemand
from actcrt_model.ac_model_f import ActorCritic
from ppo.Buffer import Buffer
import copy


class F_PPO_Online_Agent:
    def __init__(self, state_dim, action_dim, env_cfg):
        self.policy = ActorCritic(state_dim, action_dim, env_cfg)
        self.task_probs = {}
        self.heavy_build = {}
        self.order_pos = {}
        self.init_task_probs = {'ice': 1, 'ore': 0, 'rubble': 0}

    def update(self, new_params):
        self.policy.load_state_dict(new_params)

    def order_resource_pos(self, pf_info, ice_pos_list, ore_pos_list, rubble_pos_list=None):
        for p_id, f_info in pf_info.items():
            self.order_pos[p_id] = {}
            for f_id, info in f_info.items():
                pos = info['pos']
                self.order_pos[p_id][f_id] = {}
                self.order_pos[p_id][f_id]['ice'] = sorted(ice_pos_list, key=lambda x: (x[0] - pos[0]) ** 2 + (x[1] - pos[1]) ** 2)
                self.order_pos[p_id][f_id]['ore'] = sorted(ore_pos_list, key=lambda x: (x[0] - pos[0]) ** 2 + (x[1] - pos[1]) ** 2)
                self.order_pos[p_id][f_id]['rubble'] = sorted(rubble_pos_list, key=lambda x: (x[0] - pos[0]) ** 2 + (x[1] - pos[1]) ** 2)
                self.order_pos[p_id][f_id]['pos'] = pos

    def update_rubble_pos(self, pf_info, rubble_pos_list):
        for p_id, f_info in pf_info.items():
            for f_id, info in f_info.items():
                pos = info['pos']
                self.order_pos[p_id][f_id]['rubble'] = sorted(rubble_pos_list, key=lambda x: (x[0] - pos[0]) ** 2 + (x[1] - pos[1]) ** 2)

    @staticmethod
    def sample_from_task_prob(task_prob):
        rd = np.random.random()
        accum = 0
        for task in ['ice', 'ore', 'rubble']:
            accum += task_prob[task]
            if rd < accum:
                return task

    def act(self, f_obs, step, fid, max_episode_length):
        delta_demand, delta_demand_logprob, state_val = self.policy.act([f_obs])
        action = 3
        ###################################### the action choice ########################################
        if f_obs[ObsSpaceFactory.metal_dim_start] >= 100 and f_obs[ObsSpaceFactory.power_dim_start] >= 500 and step % 5 == 0:
            action = 1
            if fid not in self.heavy_build.keys():
                self.heavy_build[fid] = 0
            self.heavy_build[fid] += 1
        elif step >= max_episode_length - 62 and f_obs[ObsSpaceFactory.water_dim_start] - f_obs[ObsSpaceFactory.plant_pos_dim_start] > 20:
            action = 2
        ######################################### the demand ##################################################
        if fid not in self.task_probs.keys():
            self.task_probs[fid] = copy.deepcopy(self.init_task_probs)
        tmp = {}
        for task, increment in ActSpaceFactoryDemand.demand_id_to_increment(delta_demand[0]).items():
            tmp[task] = self.task_probs[fid][task] + increment
        if max(tmp.values()) <= 1 and min(tmp.values()) >= 0:
            self.task_probs[fid] = tmp
        return action, self.task_probs[fid], delta_demand[0], delta_demand_logprob[0], state_val[0][0]

    def reset(self):
        self.task_probs = {}
        self.heavy_build = {}


class F_PPO_Offline_Agent:
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
        }, os.path.join(self.save_dir, 'f_model.pt'))

    def load(self):
        """init maddpg using the model saved in `file`"""
        print('loading factory model .................')
        checkpoint = torch.load(os.path.join(self.save_dir, 'f_model.pt'))
        self.policy.load_state_dict(checkpoint['policy'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
