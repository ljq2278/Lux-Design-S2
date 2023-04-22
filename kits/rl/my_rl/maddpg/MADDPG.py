import logging
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F

from maddpg.UnitAgent import UnitAgent
from maddpg.Buffer import Buffer


class MADDPG:
    """A MADDPG(Multi Agent Deep Deterministic Policy Gradient) agent"""

    def __init__(self, dim_info, capacity, batch_size, actor_lr, critic_lr, res_dir):
        # sum all the dims of each agent to get input dim for critic
        # global_obs_act_dim = sum(sum(val) for val in dim_info.values())
        # create Agent(actor-critic) and replay buffer for each agent
        # self.agents = {}
        # self.buffers = {}
        self.unit_agent = UnitAgent(dim_info[0], dim_info[1], dim_info[0]+dim_info[1], actor_lr, critic_lr)
        self.unit_buffer = Buffer(capacity, dim_info[0], dim_info[1], 'cpu')
        # for agent_id, (obs_dim, act_dim) in dim_info.items():
        #     self.agents[agent_id] = UnitAgent(obs_dim, act_dim, global_obs_act_dim, actor_lr, critic_lr)
        #     self.buffers[agent_id] = Buffer(capacity, obs_dim, act_dim, 'cpu')
        self.dim_info = dim_info

        self.batch_size = batch_size
        self.res_dir = res_dir  # directory to save the training result
        # self.logger = setup_logger(os.path.join(res_dir, 'maddpg.log'))

    def add(self, obs, action, reward, next_obs, done):
        # NOTE that the experience is a dict with agent name as its key
        for unit_id in obs.keys():
            o = obs[unit_id]
            a = action[unit_id]
            if isinstance(a, int):
                # the action from env.action_space.sample() is int, we have to convert it to onehot
                a = np.eye(self.dim_info[1])[a]

            r = reward[unit_id]
            if unit_id not in next_obs.keys():
                next_o = obs[unit_id]
                d = True
            else:
                next_o = next_obs[unit_id]
                d = False
            self.unit_buffer.add(o, a, r, next_o, d)

    def sample(self, batch_size):
        """sample experience from all the agents' buffers, and collect data for network input"""
        # get the total num of transitions, these buffers should have same number of transitions
        total_num = len(self.unit_buffer)
        indices = np.random.choice(total_num, size=batch_size, replace=False)

        # NOTE that in MADDPG, we need the obs and actions of all agents
        # but only the reward and done of the current agent is needed in the calculation
        # obs, act, reward, next_obs, done, next_act = {}, {}, {}, {}, {}, {}

        o, a, r, n_o, d = self.unit_buffer.sample(indices)

        return o, a, r, n_o, d, self.unit_agent.target_action(n_o)


    def select_action(self, obs):
        actions = {}
        for agent, o in obs.items():
            o = np.array(o)
            o = torch.from_numpy(o).unsqueeze(0).float()
            a = self.unit_agent.action(o)  # torch.Size([1, action_size])
            # NOTE that the output is a tensor, convert it to int before input to the environment
            actions[agent] = a.squeeze(0).argmax().item()
            # self.logger.info(f'{agent} action: {actions[agent]}')
        return actions


    def learn(self, batch_size, gamma):
        # for agent_id, agent in self.agents.items():
        obs, act, reward, next_obs, done, next_act = self.sample(batch_size)
        # update critic
        critic_value = self.unit_agent.critic_value([obs], [act])

        # calculate target critic value
        next_target_critic_value = self.unit_agent.target_critic_value([next_obs], [next_act])
        target_value = reward + gamma * next_target_critic_value * (1 - done)

        critic_loss = F.mse_loss(critic_value, target_value.detach(), reduction='mean')
        self.unit_agent.update_critic(critic_loss)

        # update actor
        # action of the current agent is calculated using its actor
        action, logits = self.unit_agent.action(obs, model_out=True)
        act = action
        actor_loss = -self.unit_agent.critic_value([obs], [act]).mean()
        actor_loss_pse = torch.pow(logits, 2).mean()
        self.unit_agent.update_actor(actor_loss + 1e-3 * actor_loss_pse)
        # self.logger.info(f'agent{i}: critic loss: {critic_loss.item()}, actor loss: {actor_loss.item()}')


    def update_target(self, tau):
        def soft_update(from_network, to_network):
            """ copy the parameters of `from_network` to `to_network` with a proportion of tau"""
            for from_p, to_p in zip(from_network.parameters(), to_network.parameters()):
                to_p.data.copy_(tau * from_p.data + (1.0 - tau) * to_p.data)

        # for agent in self.agents.values():
        soft_update(self.unit_agent.actor, self.unit_agent.target_actor)
        soft_update(self.unit_agent.critic, self.unit_agent.target_critic)


    def save(self):
        """save actor parameters of all agents and training reward to `res_dir`"""
        torch.save({
            'actor': self.unit_agent.actor.state_dict(),
            'critic': self.unit_agent.critic.state_dict(),
            'target_actor': self.unit_agent.target_actor.state_dict(),
            'target_critic': self.unit_agent.target_critic.state_dict(),
            'actor_optimizer': self.unit_agent.actor_optimizer.state_dict(),
            'critic_optimizer': self.unit_agent.critic_optimizer.state_dict(),
        }, os.path.join(self.res_dir, 'model.pt'))
        # torch.save(
        #     self.unit_agent,  # actor parameter
        #     os.path.join(self.res_dir, 'model.pt')
        # )
        # with open(os.path.join(self.res_dir, 'rewards.pkl'), 'wb') as f:  # save training data
        #     pickle.dump({'rewards': reward}, f)

    def load(self, file):
        """init maddpg using the model saved in `file`"""
        checkpoint = torch.load(file)
        self.unit_agent.actor.load_state_dict(checkpoint['actor'])
        self.unit_agent.critic.load_state_dict(checkpoint['critic'])
        self.unit_agent.target_actor.load_state_dict(checkpoint['target_actor'])
        self.unit_agent.target_critic.load_state_dict(checkpoint['target_critic'])
        self.unit_agent.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.unit_agent.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        # self.unit_agent = torch.load(file)

