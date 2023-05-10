import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.optim import Adam
from torch.distributions import Categorical
from wrappers.obs_space_levels import ObsSpaceFactory


class ActMLPNetwork(nn.Module):
    def __init__(self, in_dim, out_dim, env_cfg, hidden_dim=32, non_linear=nn.Tanh()):
        super(ActMLPNetwork, self).__init__()
        self.obsSpaceFactory = ObsSpaceFactory(env_cfg)
        self.sfmx = nn.Softmax(dim=-1)
        self.deep_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            non_linear,
            nn.Linear(hidden_dim, hidden_dim),
            non_linear,
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        x = (x / self.obsSpaceFactory.normer).float()
        x[:, ObsSpaceFactory.pos_dim_start:ObsSpaceFactory.pos_dim_start + 2] = 0
        output = F.gumbel_softmax(self.deep_net(x), tau=5)
        return output


class CriMLPNetwork(nn.Module):
    def __init__(self, in_dim, out_dim, env_cfg, hidden_dim=32, non_linear=nn.Tanh()):
        super(CriMLPNetwork, self).__init__()
        self.obsSpaceFactory = ObsSpaceFactory(env_cfg)
        self.deep_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            non_linear,
            nn.Linear(hidden_dim, hidden_dim),
            non_linear,
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        x = (x / self.obsSpaceFactory.normer).float()
        x[:, ObsSpaceFactory.pos_dim_start:ObsSpaceFactory.pos_dim_start + 2] = 0
        ret = self.deep_net(x)
        return ret


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, env_cfg, hidden_dim=64, non_linear=nn.ReLU()):
        super(ActorCritic, self).__init__()
        self.actor = ActMLPNetwork(state_dim, action_dim, env_cfg)
        self.critic = CriMLPNetwork(state_dim, action_dim, env_cfg)

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
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        return action_logprobs, state_values, dist_entropy
