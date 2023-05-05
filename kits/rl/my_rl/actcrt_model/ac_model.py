import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.optim import Adam
from torch.distributions import Categorical
from wrappers.obs_space_levels import ObsSpaceUnit


class ActMLPNetwork(nn.Module):
    def __init__(self, in_dim, out_dim, env_cfg, hidden_dim=64, non_linear=nn.Tanh()):
        super(ActMLPNetwork, self).__init__()
        self.obsSpaceUnit = ObsSpaceUnit(env_cfg)
        self.sfmx = nn.Softmax(dim=-1)
        self.deep_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            # non_linear,
            # nn.Linear(hidden_dim, hidden_dim),
            non_linear,
            nn.Linear(hidden_dim, out_dim),
        )
        self.simple_net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, x):
        x = (x / self.obsSpaceUnit.normer).float()
        output = F.gumbel_softmax(self.simple_net(x) + self.deep_net(x), tau=2)
        # output = self.sfmx(self.simple_net(x) + self.deep_net(x))
        # output = F.gumbel_softmax(self.deep_net(x), tau=2)
        return output


class CriMLPNetwork(nn.Module):
    def __init__(self, in_dim, out_dim, env_cfg, hidden_dim=64, non_linear=nn.Tanh()):
        super(CriMLPNetwork, self).__init__()
        self.obsSpaceUnit = ObsSpaceUnit(env_cfg)
        self.p_d = nn.Parameter(torch.Tensor(3))  # ice, ore, factory
        torch.nn.init.ones_(self.p_d)
        self.scales = nn.Parameter(torch.Tensor(5))
        torch.nn.init.ones_(self.scales)
        self.simple_net = nn.Sequential(
            nn.Linear(7, 1),
        )
        self.deep_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            # non_linear,
            # nn.Linear(hidden_dim, hidden_dim),
            non_linear,
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        x = (x / self.obsSpaceUnit.normer).float()
        # v_ice = torch.exp(
        #     -(x[:, ObsSpaceUnit.:31] ** 2 + x[:, 31:32] ** 2) / (2 * self.p_d[0] ** 2))
        # v_ore = torch.exp(
        #     -(x[:, 32:33] ** 2 + x[:, 33:34] ** 2) / (2 * self.p_d[1] ** 2))
        v_target = torch.exp(
            -(x[:, ObsSpaceUnit.target_pos_start:ObsSpaceUnit.target_pos_start + 1] ** 2 +
              x[:, ObsSpaceUnit.target_pos_start + 1:ObsSpaceUnit.target_pos_start + 2] ** 2) / (2 * self.p_d[2] ** 2))
        v_factory = torch.exp(
            -(x[:, ObsSpaceUnit.target_factory_pos_start:ObsSpaceUnit.target_factory_pos_start + 1] ** 2 +
              x[:, ObsSpaceUnit.target_factory_pos_start + 1:ObsSpaceUnit.target_factory_pos_start + 2] ** 2) / (2 * self.p_d[2] ** 2))

        # v_factory = self.p_s * torch.exp(
        #     -((x[30] - self.p_e[0]) ** 2 + (x[31] - self.p_e[1]) ** 2) / (2 * self.p_d ** 2))
        simple_input = torch.concat([
            x[:, ObsSpaceUnit.power_dim_start:ObsSpaceUnit.power_dim_start + 1],
            x[:, ObsSpaceUnit.ice_dim_start:ObsSpaceUnit.ice_dim_start + 1],
            x[:, ObsSpaceUnit.ore_dim_start:ObsSpaceUnit.ore_dim_start + 1],
            v_factory,
            v_target,
            x[:, ObsSpaceUnit.task_type_start:ObsSpaceUnit.task_type_start + 1],
            x[:, ObsSpaceUnit.transfered_start:ObsSpaceUnit.transfered_start + 1]
        ], dim=1)
        ret = self.simple_net(simple_input) + self.deep_net(x)
        # input = torch.concat([x, v_ice, v_ore, v_factory], dim=1)
        # ret = self.deep_net(x)
        return ret


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, env_cfg, hidden_dim=64, non_linear=nn.ReLU()):
        super(ActorCritic, self).__init__()
        self.actor = ActMLPNetwork(state_dim, action_dim, env_cfg)
        self.critic = CriMLPNetwork(state_dim, action_dim, env_cfg)
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
