from copy import deepcopy
from typing import List
import os
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.optim import Adam
from torch.distributions import Categorical
import numpy as np


class UnitAgent:
    """Agent that can interact with environment from pettingzoo"""

    def __init__(self, obs_dim, act_dim, unit_buffer, save_dir,
                 actor_lr=0.0001, critic_lr=0.0001, total_lr=0.0001, debug=False):
        self.unit_buffer = unit_buffer
        self.actor = ActMLPNetwork(obs_dim, act_dim)
        self.critic = CriMLPNetwork(obs_dim, 1)
        # self.actor_critic = ActCriMLPNetwork(obs_dim, act_dim)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=critic_lr)
        # self.actor_critic_optimizer = Adam(self.actor_critic.parameters(), lr=total_lr)
        self.save_dir = save_dir
        self.debug = debug
        # self.test = 0

    def update_actor(self, loss):
        self.actor_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        if self.debug:
            print('update actor start .......')
            self.get_actor_params()
        self.actor_optimizer.step()
        if self.debug:
            print('....... update actor over')
            self.get_actor_params()

    def update_critic(self, loss):
        self.critic_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()

    def update_actor_critic(self, loss):
        self.actor_critic_optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.actor_critic_optimizer.step()

    def get_action(self, obs, debug=False):
        o = np.array(obs)
        o = torch.from_numpy(o).unsqueeze(0).float()
        dist = self.actor(o)
        a = dist.sample()  # torch.Size([1, action_size])
        return a.squeeze(0)

    def learn(self, gamma):
        # for agent_id, agent in self.agents.items():
        print('on learning ...............')
        obs, act, reward, next_obs, done = self.unit_buffer.get_replays()
        actor_dist = self.actor(torch.from_numpy(np.array(obs)).float())
        critic_value = self.critic(torch.from_numpy(np.array(obs)).float())
        next_critic_value = self.critic(torch.from_numpy(np.array(next_obs)).float()).detach()
        # _, next_critic_value = self.actor_critic(torch.from_numpy(next_obs).float())

        ts_act = torch.FloatTensor(np.array(act))
        ts_reward = torch.FloatTensor(np.array(reward)).unsqueeze(1)
        ts_live = torch.FloatTensor(1 - np.array(done)).unsqueeze(1)

        Q = ts_reward + gamma * next_critic_value * ts_live
        advantage = Q - critic_value

        actor_loss = -(actor_dist.log_prob(ts_act) * advantage.detach()).mean()
        critic_loss = 0.5 * advantage.pow(2).mean()
        entropy_loss = -0.0 * actor_dist.entropy().mean()
        self.update_actor(actor_loss + entropy_loss)
        self.update_critic(critic_loss)
        self.unit_buffer.clear()
        # self.update_actor_critic(actor_loss + critic_loss + entropy_loss)
        print('............... end learning ')

    def get_actor_params(self):
        print('actor params: ......................... start')
        for param in self.actor.parameters():
            print(param.data)
        print('actor params: ......................... over')

    def get_critic_params(self):
        print('critic params: ......................... start')
        for param in self.critic.parameters():
            print(param.data)
        print('critic params: ......................... over')

    def save(self):
        """save actor parameters of all agents and training reward to `res_dir`"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            # 'actor_critic': self.actor_critic.state_dict(),
            # 'target_actor': self.unit_agent.target_actor.state_dict(),
            # 'target_critic': self.unit_agent.target_critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            # 'actor_critic_optimizer': self.actor_critic_optimizer.state_dict(),
        }, os.path.join(self.save_dir, 'model.pt'))

    def load(self):
        """init maddpg using the model saved in `file`"""
        print('loading model .................')
        checkpoint = torch.load(os.path.join(self.save_dir, 'model.pt'))
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        # self.actor_critic.load_state_dict(checkpoint['actor_critic'])
        # self.unit_agent.target_actor.load_state_dict(checkpoint['target_actor'])
        # self.unit_agent.target_critic.load_state_dict(checkpoint['target_critic'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        # self.actor_critic_optimizer.load_state_dict(checkpoint['actor_critic_optimizer'])
        # self.unit_agent = torch.load(file)


class ActCriMLPNetwork(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=64, non_linear=nn.ReLU()):
        super(ActCriMLPNetwork, self).__init__()
        # self.sfmx = nn.Softmax(dim=1)
        self.shared_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            non_linear,
            nn.Linear(hidden_dim, hidden_dim),
            non_linear,
            nn.Linear(hidden_dim, hidden_dim),
            non_linear,
        ).apply(self.init)
        self.actor_net = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.Softmax(dim=1),
        ).apply(self.init)
        self.critic_net = nn.Sequential(
            nn.Linear(hidden_dim, 1),
        ).apply(self.init)

    @staticmethod
    def init(m):
        """init parameter of the module"""
        gain = nn.init.calculate_gain('relu')
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=gain)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        shared_output = self.shared_net(x)
        actor_ouput = self.actor_net(shared_output)
        critic_output = self.critic_net(shared_output)
        actor_dist = Categorical(actor_ouput)
        return actor_dist, critic_output


class ActMLPNetwork(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=64, non_linear=nn.ReLU()):
        super(ActMLPNetwork, self).__init__()
        self.sfmx = nn.Softmax(dim=1)
        self.deep_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            non_linear,
            # nn.Linear(hidden_dim, hidden_dim),
            # non_linear,
            nn.Linear(hidden_dim, out_dim),
            # nn.Softmax(dim=1),
        ).apply(self.init)
        self.simple_net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            # nn.Softmax(dim=1),
        ).apply(self.init)

    @staticmethod
    def init(m):
        """init parameter of the module"""
        gain = nn.init.calculate_gain('relu')
        if isinstance(m, nn.Linear):
            # torch.nn.init.xavier_uniform_(m.weight, gain=gain)
            # m.bias.data.fill_(0.01)
            torch.nn.init.zeros_(m.weight)
            m.bias.data.fill_(0.0)

    def forward(self, x):
        # output = F.gumbel_softmax(self.simple_net(x) + self.deep_net(x), tau=2)
        output = self.sfmx(self.simple_net(x) + self.deep_net(x))
        dist = Categorical(output)
        return dist


class CriMLPNetwork(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=64, non_linear=nn.ReLU()):
        super(CriMLPNetwork, self).__init__()
        self.p_d = nn.Parameter(torch.Tensor(3))  # ice, ore, factory
        torch.nn.init.ones_(self.p_d)
        # self.p_s = nn.Parameter(torch.Tensor(3))
        # torch.nn.init.ones_(self.p_s)
        # self.p_b = nn.Parameter(torch.Tensor(3))
        # torch.nn.init.ones_(self.p_b)
        self.scales = nn.Parameter(torch.Tensor(5))
        torch.nn.init.ones_(self.scales)

        self.simple_net = nn.Sequential(
            nn.Linear(7, 1),
        ).apply(self.init_simple)

        self.deep_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            non_linear,
            nn.Linear(hidden_dim, hidden_dim),
            non_linear,
            nn.Linear(hidden_dim, out_dim),
            # non_linear,
        ).apply(self.init_deep)

    @staticmethod
    def init_simple(m):
        """init parameter of the module"""
        if isinstance(m, nn.Linear):
            torch.nn.init.ones_(m.weight)
            torch.nn.init.zeros_(m.bias)

    @staticmethod
    def init_deep(m):
        """init parameter of the module"""
        gain = nn.init.calculate_gain('relu')
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=gain)
            m.bias.data.fill_(0.01)
            # torch.nn.init.zeros_(m.weight)
            # torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        v_ice = torch.exp(
            -(x[:, 30:31] ** 2 + x[:, 31:32] ** 2) / (2 * self.p_d[0] ** 2))
        v_ore = torch.exp(
            -(x[:, 32:33] ** 2 + x[:, 33:34] ** 2) / (2 * self.p_d[1] ** 2))
        v_factory = torch.exp(
            -(x[:, 34:35] ** 2 + x[:, 35:36] ** 2) / (2 * self.p_d[2] ** 2))
        # v_factory = self.p_s * torch.exp(
        #     -((x[30] - self.p_e[0]) ** 2 + (x[31] - self.p_e[1]) ** 2) / (2 * self.p_d ** 2))
        simple_input = torch.concat([x[:, 2:3], x[:, 3:4], x[:, 4:5], x[:, 38:39], v_ice, v_ore, v_factory], dim=1)
        ret = self.simple_net(simple_input) + self.deep_net(x)
        return ret
