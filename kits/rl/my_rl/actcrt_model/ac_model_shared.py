import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.optim import Adam
from torch.distributions import Categorical
from wrappers.obs_space_conv import ObsSpace, ObsSpaceStat
from .conv_nets import UActNet, FActNet, ValueNet, BaseNet, Decoder


class ActMLPNetwork(nn.Module):
    def __init__(self, in_dim, f_out_dim, u_out_dim, env_cfg, base_net, gumbel_softmax_tau=8):
        super(ActMLPNetwork, self).__init__()
        self.gumbel_softmax_tau = gumbel_softmax_tau
        self.obs_space = ObsSpace(env_cfg)
        self.f_deep_net = FActNet(in_dim, f_out_dim, base_net)
        self.u_deep_net = UActNet(in_dim, u_out_dim, base_net)

    def forward(self, x, device='cpu'):
        if device == 'cpu':
            normer = torch.unsqueeze(torch.unsqueeze(torch.tensor(self.obs_space.normer), 1), 1)
        else:
            normer = torch.unsqueeze(torch.unsqueeze(torch.tensor(self.obs_space.normer).cuda(), 1), 1)
        x = (x / normer).float()
        f_output = F.gumbel_softmax(self.f_deep_net(x), dim=1, tau=self.gumbel_softmax_tau)
        u_output = F.gumbel_softmax(self.u_deep_net(x), dim=1, tau=self.gumbel_softmax_tau)
        return f_output, u_output


class CriMLPNetwork(nn.Module):
    def __init__(self, in_dim, out_dim, env_cfg, base_net, hidden_dim=32, non_linear=nn.Tanh()):
        super(CriMLPNetwork, self).__init__()
        self.obs_space = ObsSpace(env_cfg)
        self.obs_space_stat = ObsSpaceStat()
        self.deep_net = ValueNet(in_dim, out_dim, base_net, 128 * (self.obs_space.env_cfg.map_size // 4) * (self.obs_space.env_cfg.map_size // 4))

    def forward(self, x, x_stat, device='cpu'):
        if device == 'cpu':
            normer = torch.unsqueeze(torch.unsqueeze(torch.tensor(self.obs_space.normer), 1), 1)
        else:
            normer = torch.unsqueeze(torch.unsqueeze(torch.tensor(self.obs_space.normer).cuda(), 1), 1)
        x = (x / normer).float()
        h, ret = self.deep_net(x)
        return h, ret


class ActorCritic(nn.Module):
    def __init__(self, state_dim, f_action_dim, u_action_dim, env_cfg, gumbel_softmax_tau=8):
        super(ActorCritic, self).__init__()
        self.base_net = BaseNet(state_dim)
        self.decoder = Decoder(state_dim)
        self.actor = ActMLPNetwork(state_dim, f_action_dim, u_action_dim, env_cfg, self.base_net, gumbel_softmax_tau)
        self.critic = CriMLPNetwork(state_dim, 1, env_cfg, self.base_net)

    def forward(self):
        raise NotImplementedError

    def act(self, state, state_stat, device='gpu'):
        with torch.no_grad():
            state = torch.FloatTensor(state)
            state_stat = torch.FloatTensor(state_stat)
            if device == 'gpu':
                state = state.cuda()
                state_stat = state_stat.cuda()
            h, state_val = self.critic(state, state_stat, device=device)
            f_action_probs, u_action_probs = self.actor(state, device=device)
            f_dist, u_dist = Categorical(torch.permute(f_action_probs, (0, 2, 3, 1))), Categorical(torch.permute(u_action_probs, (0, 2, 3, 1)))
            f_action, u_action = f_dist.sample(), u_dist.sample()
            f_action_logprob, u_action_logprob = f_dist.log_prob(f_action), u_dist.log_prob(u_action)
            return state_val.tolist(), f_action.tolist(), f_action_logprob.tolist(), u_action.tolist(), u_action_logprob.tolist()

    def evaluate(self, state, state_stat, f_action, u_action):
        h, state_values = self.critic(state, state_stat, device='gpu')
        f_action_probs, u_action_probs = self.actor(state, device='gpu')
        f_dist, u_dist = Categorical(torch.permute(f_action_probs, (0, 2, 3, 1))), Categorical(torch.permute(u_action_probs, (0, 2, 3, 1)))
        f_action_logprobs, u_action_logprobs = f_dist.log_prob(f_action), u_dist.log_prob(u_action)
        f_dist_entropy, u_dist_entropy = f_dist.entropy(), u_dist.entropy()
        return state_values, f_action_logprobs, f_dist_entropy, u_action_logprobs, u_dist_entropy, h
