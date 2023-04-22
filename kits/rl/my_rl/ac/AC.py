import logging
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F

from ac.UnitAgent import UnitAgent
from ac.UnitBuffer import Buffer


class AC:
    """A MADDPG(Multi Agent Deep Deterministic Policy Gradient) agent"""

    def __init__(self, res_dir, unit_agent, unit_buffer):

        self.unit_agent = unit_agent
        self.unit_buffer = unit_buffer
        self.res_dir = res_dir  # directory to save the training result
        # self.logger = setup_logger(os.path.join(res_dir, 'maddpg.log'))

    def add(self, obs, action, reward, next_obs, done):
        # NOTE that the experience is a dict with agent name as its key
        for unit_id in obs.keys():
            o = obs[unit_id]
            a = action[unit_id]
            r = reward[unit_id]
            if unit_id not in next_obs.keys():
                next_o = obs[unit_id]
                d = True
            else:
                next_o = next_obs[unit_id]
                d = False
            self.unit_buffer.add(o, a, r, next_o, d)

    def batch(self):
        o, a, r, n_o, d = self.unit_buffer.get()
        return o, a, r, n_o, d

    def get_action(self, obs):
        actions = {}
        for agent, o in obs.items():
            o = np.array(o)
            o = torch.from_numpy(o).unsqueeze(0).float()
            dist = self.unit_agent.actor(o)
            a = dist.sample()  # torch.Size([1, action_size])
            actions[agent] = a.squeeze(0)
        return actions

    def learn(self,  gamma):
        # for agent_id, agent in self.agents.items():
        print('on learning ...............')
        obs, act, reward, next_obs, done = self.batch()
        actor_dist = self.unit_agent.actor(torch.from_numpy(obs).float())
        critic_value = self.unit_agent.critic(torch.from_numpy(obs).float())
        next_critic_value = self.unit_agent.critic(torch.from_numpy(next_obs).float())

        ts_act = torch.FloatTensor(act)
        ts_reward = torch.FloatTensor(reward).unsqueeze(1)
        ts_live = torch.FloatTensor(1 - done).unsqueeze(1)

        Q = ts_reward + gamma * next_critic_value * ts_live
        advantage = Q - critic_value

        actor_loss = -(actor_dist.log_prob(ts_act) * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()

        self.unit_agent.update_actor(actor_loss)
        self.unit_agent.update_critic(critic_loss)
        print('............... end learning ')

    def save(self):
        """save actor parameters of all agents and training reward to `res_dir`"""
        torch.save({
            'actor': self.unit_agent.actor.state_dict(),
            'critic': self.unit_agent.critic.state_dict(),
            # 'target_actor': self.unit_agent.target_actor.state_dict(),
            # 'target_critic': self.unit_agent.target_critic.state_dict(),
            'actor_optimizer': self.unit_agent.actor_optimizer.state_dict(),
            'critic_optimizer': self.unit_agent.critic_optimizer.state_dict(),
        }, os.path.join(self.res_dir, 'model.pt'))

    def load(self, file):
        """init maddpg using the model saved in `file`"""
        checkpoint = torch.load(file)
        self.unit_agent.actor.load_state_dict(checkpoint['actor'])
        self.unit_agent.critic.load_state_dict(checkpoint['critic'])
        # self.unit_agent.target_actor.load_state_dict(checkpoint['target_actor'])
        # self.unit_agent.target_critic.load_state_dict(checkpoint['target_critic'])
        self.unit_agent.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.unit_agent.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        # self.unit_agent = torch.load(file)
