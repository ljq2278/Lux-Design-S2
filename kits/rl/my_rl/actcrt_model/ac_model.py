import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.optim import Adam
from torch.distributions import Categorical


class ActMLPNetwork(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=64, non_linear=nn.ReLU()):
        super(ActMLPNetwork, self).__init__()
        self.sfmx = nn.Softmax(dim=-1)
        self.deep_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            non_linear,
            nn.Linear(hidden_dim, out_dim),
        )
        self.simple_net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
        )
    def forward(self, x):
        # output = F.gumbel_softmax(self.simple_net(x) + self.deep_net(x), tau=2)
        output = self.sfmx(self.simple_net(x) + self.deep_net(x))
        return output

class CriMLPNetwork(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=64, non_linear=nn.ReLU()):
        super(CriMLPNetwork, self).__init__()
        self.p_d = nn.Parameter(torch.Tensor(3))  # ice, ore, factory
        torch.nn.init.ones_(self.p_d)
        self.scales = nn.Parameter(torch.Tensor(5))
        torch.nn.init.ones_(self.scales)
        self.simple_net = nn.Sequential(
            nn.Linear(7, 1),
        )
        self.deep_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            non_linear,
            nn.Linear(hidden_dim, hidden_dim),
            non_linear,
            nn.Linear(hidden_dim, 1),
        )

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

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64,non_linear=nn.ReLU()):
        super(ActorCritic, self).__init__()
        self.actor = ActMLPNetwork(state_dim, action_dim)
        self.critic = CriMLPNetwork(state_dim, action_dim)
        # self.actor = nn.Sequential(
        #     nn.Linear(state_dim, 64),
        #     nn.Tanh(),
        #     nn.Linear(64, 64),
        #     nn.Tanh(),
        #     nn.Linear(64, action_dim),
        #     nn.Softmax(dim=-1)
        # )
        # # critic
        # self.critic = nn.Sequential(
        #     nn.Linear(state_dim, 64),
        #     nn.Tanh(),
        #     nn.Linear(64, 64),
        #     nn.Tanh(),
        #     nn.Linear(64, 1)
        # )

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        state = torch.FloatTensor(state)
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)
        return action.tolist(), action_logprob.tolist(), state_val.tolist()

    def evaluate(self, state, action):
        # state = torch.FloatTensor(state)
        # action = torch.FloatTensor(action)
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        return action_logprobs, state_values, dist_entropy


