import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.optim import Adam
from torch.distributions import Categorical
from wrappers.obs_space_conv import ObsSpace
from .unet import UNet


class ActMLPNetwork(nn.Module):
    def __init__(self, in_dim, f_out_dim, u_out_dim, env_cfg, hidden_dim=32, non_linear=nn.Tanh()):
        super(ActMLPNetwork, self).__init__()
        self.obs_space = ObsSpace(env_cfg)
        self.f_deep_net = UNet(in_dim, f_out_dim)
        self.u_deep_net = UNet(in_dim, u_out_dim)

    def forward(self, x, device='cpu'):
        if device == 'cpu':
            normer = torch.unsqueeze(torch.unsqueeze(torch.tensor(self.obs_space.normer), 1), 1)
        else:
            normer = torch.unsqueeze(torch.unsqueeze(torch.tensor(self.obs_space.normer).cuda(), 1), 1)
        x = (x / normer).float()
        f_output = F.gumbel_softmax(self.f_deep_net(x), dim=1, tau=8)
        u_output = F.gumbel_softmax(self.u_deep_net(x), dim=1, tau=8)
        return f_output, u_output


class CriMLPNetwork(nn.Module):
    def __init__(self, in_dim, out_dim, env_cfg, hidden_dim=32, non_linear=nn.Tanh()):
        super(CriMLPNetwork, self).__init__()
        self.obs_space = ObsSpace(env_cfg)
        self.deep_net = UNet(in_dim, out_dim)
        self.fc = nn.Linear(self.obs_space.env_cfg.map_size * self.obs_space.env_cfg.map_size * out_dim, 1)

    def forward(self, x, device='cpu'):
        if device == 'cpu':
            normer = torch.unsqueeze(torch.unsqueeze(torch.tensor(self.obs_space.normer), 1), 1)
        else:
            normer = torch.unsqueeze(torch.unsqueeze(torch.tensor(self.obs_space.normer).cuda(), 1), 1)
        x = (x / normer).float()
        x = self.deep_net(x)
        ret = torch.sum(x.reshape([-1, 1, 1 * self.obs_space.env_cfg.map_size * self.obs_space.env_cfg.map_size]), axis=-1)
        return ret


class ActorCritic(nn.Module):
    def __init__(self, state_dim, f_action_dim, u_action_dim, env_cfg, hidden_dim=64, non_linear=nn.ReLU()):
        super(ActorCritic, self).__init__()
        self.actor = ActMLPNetwork(state_dim, f_action_dim, u_action_dim, env_cfg)
        self.critic = CriMLPNetwork(state_dim, 1, env_cfg)

    def forward(self):
        raise NotImplementedError

    def act(self, state, device='gpu'):
        with torch.no_grad():
            state = torch.FloatTensor(state)
            if device == 'gpu':
                state = state.cuda()
            state_val = self.critic(state, device=device)
            f_action_probs, u_action_probs = self.actor(state, device=device)
            f_dist, u_dist = Categorical(torch.permute(f_action_probs, (0, 2, 3, 1))), Categorical(torch.permute(u_action_probs, (0, 2, 3, 1)))
            f_action, u_action = f_dist.sample(), u_dist.sample()
            f_action_logprob, u_action_logprob = f_dist.log_prob(f_action), u_dist.log_prob(u_action)
            return state_val.tolist(), f_action.tolist(), f_action_logprob.tolist(), u_action.tolist(), u_action_logprob.tolist(),

    def evaluate(self, state, f_action, u_action):
        state_values = self.critic(state, device='gpu')
        f_action_probs, u_action_probs = self.actor(state, device='gpu')
        f_dist, u_dist = Categorical(torch.permute(f_action_probs, (0, 2, 3, 1))), Categorical(torch.permute(u_action_probs, (0, 2, 3, 1)))
        f_action_logprobs, u_action_logprobs = f_dist.log_prob(f_action), u_dist.log_prob(u_action)
        f_dist_entropy, u_dist_entropy = f_dist.entropy(), u_dist.entropy()
        return state_values, f_action_logprobs, f_dist_entropy, u_action_logprobs, u_dist_entropy
