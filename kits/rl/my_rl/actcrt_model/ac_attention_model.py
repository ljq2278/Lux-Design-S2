import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.optim import Adam
from torch.distributions import Categorical
from wrappers.obs_space_levels import ObsSpaceUnit
import numpy as np


class ActMLPNetwork(nn.Module):
    def __init__(self, in_dim, out_dim, env_cfg, num_bins=1100, embedding_dim=8, mn=-1, mx=300, hidden_dim=64, non_linear=nn.Tanh()):
        super(ActMLPNetwork, self).__init__()
        self.obsSpaceUnit = ObsSpaceUnit(env_cfg)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.embedding_dim = embedding_dim
        self.num_bins = num_bins
        self.mn = mn
        self.mx = mx
        self.embeddings = nn.ModuleList([nn.Embedding(num_bins, embedding_dim) for _ in range(in_dim)])
        self.attn_net = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=4)
        self.linear = nn.Linear(in_dim * embedding_dim, out_dim)
        self.sfmx = nn.Softmax(dim=-1)

    def discretize(self, x):
        bins = torch.linspace(self.mn, self.mx, self.num_bins)
        x = x.permute(1, 0).contiguous()
        binned = torch.stack([torch.bucketize(x[i, :], bins) for i in range(x.shape[0])], dim=1)
        return binned

    def forward(self, x):
        x = torch.clamp(x, self.mn, self.mx)
        x_discrete = self.discretize(x)
        x_embedded = torch.stack([self.embeddings[i](x_discrete[:, i]) for i in range(x.shape[1])], dim=1)
        atte_out, _ = self.attn_net(x_embedded, x_embedded, x_embedded)
        atte_out_rs = atte_out.reshape([-1, self.embedding_dim * self.in_dim])
        ret = self.linear(atte_out_rs)
        ret_prob = self.sfmx(ret)
        return ret_prob


class CriMLPNetwork(nn.Module):
    def __init__(self, in_dim, out_dim, env_cfg, num_bins=1100, embedding_dim=8, mn=-1, mx=300, hidden_dim=64, non_linear=nn.Tanh()):
        super(CriMLPNetwork, self).__init__()
        self.obsSpaceUnit = ObsSpaceUnit(env_cfg)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.embedding_dim = embedding_dim
        self.num_bins = num_bins
        self.mn = mn
        self.mx = mx
        self.embeddings = nn.ModuleList([nn.Embedding(num_bins, embedding_dim) for _ in range(in_dim)])
        self.attn_net = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=1)
        self.linear = nn.Linear(in_dim * embedding_dim, 1)

    def discretize(self, x):
        bins = torch.linspace(self.mn, self.mx, self.num_bins)
        x = x.permute(1, 0).contiguous()
        binned = torch.stack([torch.bucketize(x[i, :], bins) for i in range(x.shape[0])], dim=1)
        return binned

    def forward(self, x):
        x = torch.clamp(x, self.mn, self.mx)
        x_discrete = self.discretize(x)
        x_embedded = torch.stack([self.embeddings[i](x_discrete[:, i]) for i in range(x.shape[1])], dim=1)
        atte_out, _ = self.attn_net(x_embedded, x_embedded, x_embedded)
        atte_out_rs = atte_out.reshape([-1, self.embedding_dim * self.in_dim])
        ret = self.linear(atte_out_rs)
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
