import os
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.optim import Adam
from torch.distributions import Categorical
import numpy as np
from collections import OrderedDict
from wrappers.obs_space_conv import ObsSpace
# from wrappers.act_space_levels import ActSpaceFactoryDemand
# from actcrt_model.ac_model_conv import ActorCritic
from actcrt_model.ac_model_shared import ActorCritic
from ppo_conv.Buffer import Buffer
import copy


class CentralAgent:
    def __init__(self, state_dim, f_action_dim, u_action_dim, env_cfg, lr, gumbel_softmax_tau=8, save_dir='./', is_cuda=True):
        self.save_dir = save_dir
        if is_cuda:
            self.policy = ActorCritic(state_dim, f_action_dim, u_action_dim, env_cfg, gumbel_softmax_tau).cuda()
        else:
            self.policy = ActorCritic(state_dim, f_action_dim, u_action_dim, env_cfg, gumbel_softmax_tau)
        if lr is not None:
            self.optimizer = torch.optim.Adam([
                {'params': list(self.policy.actor.u_deep_net.outc.parameters()), 'lr': lr['actor_u_lr']},
                {'params': list(self.policy.actor.f_deep_net.outc.parameters()), 'lr': lr['actor_f_lr']},
                {'params': list(self.policy.actor.u_deep_net.mask_outc.parameters()), 'lr': lr['actor_u_mask_lr']},
                {'params': list(self.policy.actor.f_deep_net.mask_outc.parameters()), 'lr': lr['actor_f_mask_lr']},
                {'params': self.policy.critic.deep_net.fc.parameters(), 'lr': lr['critic_lr']},
                {'params': self.policy.base_net.parameters(), 'lr': lr['encoder_lr']},
                {'params': self.policy.decoder.parameters(), 'lr': lr['decoder_lr']}
            ])

    def save(self):
        """save actor parameters of all agents and training reward to `res_dir`"""
        torch.save({
            'policy': self.policy.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, os.path.join(self.save_dir, 'model.pt'))

    def load(self):
        """init maddpg using the model saved in `file`"""
        print('loading model .................')
        checkpoint = torch.load(os.path.join(self.save_dir, 'model.pt'))
        self.policy.load_state_dict(checkpoint['policy'])
        # self.optimizer.load_state_dict(checkpoint['optimizer'])


class CentralOnlineAgent(CentralAgent):
    def __init__(self, state_dim, f_action_dim, u_action_dim, env_cfg, lr=None, gumbel_softmax_tau=8, save_dir='./', is_cuda=True):
        super().__init__(state_dim, f_action_dim, u_action_dim, env_cfg, lr, gumbel_softmax_tau, save_dir, is_cuda)

    def update(self, new_params):
        self.policy.load_state_dict(new_params)

    def soft_update(self, new_params, tau=0.5):
        soft_new_params = OrderedDict()
        for k, v in self.policy.state_dict().items():
            soft_new_params[k] = (1 - tau) * v.to('cpu') + tau * new_params[k].to('cpu')
        self.policy.load_state_dict(soft_new_params)


class CentralOfflineAgent(CentralAgent):
    def __init__(self, state_dim, f_action_dim, u_action_dim, env_cfg, lr, gumbel_softmax_tau=8, eps_clip=0.2, save_dir='./', is_cuda=True):
        super().__init__(state_dim, f_action_dim, u_action_dim, env_cfg, lr, gumbel_softmax_tau, save_dir, is_cuda)
        self.eps_clip = eps_clip
        self.obs_space = ObsSpace(env_cfg)
        self.mseLoss = nn.MSELoss()

    def update_and_get_new_param2(self, train_data, K_epochs, bz, log_writer, online_agent_update_time, entropy_loss_factor,
                                  v_loss_factor, f_loss_factor, u_loss_factor, ed_loss_factor, l1_factor, l2_factor):
        # Monte Carlo estimate of returns
        # Optimize policy for K epochs
        tt_loss = {
            'v_loss': 0,
            'f_loss': 0,
            'u_loss': 0,
            'ed_loss': 0,
        }
        tt = 0
        self.policy.to('cuda')
        pid_num = len(train_data)
        tt_old_states, tt_old_states_stat, tt_old_state_vals, tt_old_f_actions, tt_old_f_logprobs, tt_old_u_actions, tt_old_u_logprobs, tt_old_rewards, tt_old_done, tt_advantages = [], [], [], [], [], [], [], [], [], []
        for i in range(0, pid_num):
            tt_old_states += train_data[i][0]
            tt_old_states_stat += train_data[i][1]
            tt_old_state_vals += train_data[i][2]
            tt_old_f_actions += train_data[i][3]
            tt_old_f_logprobs += train_data[i][4]
            tt_old_u_actions += train_data[i][5]
            tt_old_u_logprobs += train_data[i][6]
            tt_old_rewards += train_data[i][7]
            tt_old_done += train_data[i][8]
            tt_advantages += train_data[i][9]
        for epochs_i in range(K_epochs):
            print('train_epochs: ', epochs_i)
            permute_list = np.random.permutation(len(tt_old_states))
            for i in range(0, len(tt_old_states), bz):
                old_states, old_states_stat, old_state_vals, old_f_actions, old_f_logprobs, old_u_actions, old_u_logprobs, old_rewards, old_done, advantages \
                    = torch.Tensor(np.array(tt_old_states)[permute_list[i:i + bz]]).cuda(), \
                      torch.Tensor(np.array(tt_old_states_stat)[permute_list[i:i + bz]]).cuda(), \
                      torch.Tensor(np.array(tt_old_state_vals)[permute_list[i:i + bz]]).cuda(), \
                      torch.Tensor(np.array(tt_old_f_actions)[permute_list[i:i + bz]]).cuda(), \
                      torch.Tensor(np.array(tt_old_f_logprobs)[permute_list[i:i + bz]]).cuda(), \
                      torch.Tensor(np.array(tt_old_u_actions)[permute_list[i:i + bz]]).cuda(), \
                      torch.Tensor(np.array(tt_old_u_logprobs)[permute_list[i:i + bz]]).cuda(), \
                      torch.Tensor(np.array(tt_old_rewards)[permute_list[i:i + bz]]).cuda(), \
                      torch.Tensor(np.array(tt_old_done)[permute_list[i:i + bz]]).cuda(), \
                      torch.Tensor(np.array(tt_advantages)[permute_list[i:i + bz]]).cuda()
                old_f_masks, old_u_masks = old_states[:, self.obs_space.f_pos_dim_start, :, :].cuda(), old_states[:, self.obs_space.u_pos_dim_start, :, :].cuda()
                state_values, f_logprobs, f_dist_entropy, u_logprobs, u_dist_entropy, hidden = self.policy.evaluate(old_states, old_states_stat, old_f_actions, old_u_actions)
                # match state_values tensor dimensions with rewards tensor
                state_values = torch.squeeze(state_values)
                # Finding the ratio (pi_theta / pi_theta__old)
                f_ratios, u_ratios = torch.exp(f_logprobs - old_f_logprobs), torch.exp(u_logprobs - old_u_logprobs)
                # Finding Surrogate Loss
                us_advantages = advantages.unsqueeze(dim=1).unsqueeze(dim=1)
                f_surr1, u_surr1 = f_ratios * us_advantages, u_ratios * us_advantages
                f_surr2, u_surr2 = torch.clamp(f_ratios, 1 - self.eps_clip, 1 + self.eps_clip) * us_advantages, \
                                   torch.clamp(u_ratios, 1 - self.eps_clip, 1 + self.eps_clip) * us_advantages
                # final loss of clipped objective PPO
                obs_normer = torch.FloatTensor(self.policy.critic.obs_space.normer).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(old_states.shape).cuda()

                v_loss = v_loss_factor * self.mseLoss(state_values, old_rewards)

                f_one_hot = torch.nn.functional.one_hot(old_f_actions.reshape([-1]).to(torch.int64), self.policy.actor.f_deep_net.n_classes)
                f_tensor_one_hot = f_one_hot.reshape([old_f_actions.shape[0], old_f_actions.shape[1], old_f_actions.shape[2], -1]).permute(0, 3, 1, 2)
                f_atten = torch.clamp(self.policy.actor.f_deep_net.get_mask(old_states / obs_normer, f_tensor_one_hot.cuda()), 0.04, 1)
                f_loss = f_loss_factor * (-torch.min(f_surr1, f_surr2)) * old_f_masks * f_atten
                f_entropy_loss = -entropy_loss_factor * f_dist_entropy

                u_one_hot = torch.nn.functional.one_hot(old_u_actions.reshape([-1]).to(torch.int64), self.policy.actor.u_deep_net.n_classes)
                u_tensor_one_hot = u_one_hot.reshape([old_u_actions.shape[0], old_u_actions.shape[1], old_u_actions.shape[2], -1]).permute(0, 3, 1, 2)
                u_atten = torch.clamp(self.policy.actor.u_deep_net.get_mask(old_states / obs_normer, u_tensor_one_hot.cuda()), 0.04, 1)
                u_loss = u_loss_factor * (-torch.min(u_surr1, u_surr2)) * old_u_masks * u_atten
                u_entropy_loss = -entropy_loss_factor * u_dist_entropy

                if i == 0:
                    print('#################################### u_tensor_one_hot start #######################################')
                    # for itm in u_atten[0, 0, :, :].tolist():
                    #     print(['%.03f' % x for x in itm])
                    # print('#################################### unit_pos start #######################################')
                    # for itm in old_states[0, self.policy.actor.obs_space.u_pos_dim_start, :, :].tolist():
                    #     print(['%.03f' % x for x in itm])
                    print('ori_units atten_units: ', torch.sum(old_states[:, self.policy.actor.obs_space.u_pos_dim_start, :, :]).tolist(),
                          torch.sum(u_atten[:, 0, :, :] * old_states[:, self.policy.actor.obs_space.u_pos_dim_start, :, :]).tolist())
                    print('#################################### f_tensor_one_hot mid #######################################')
                    # for itm in f_atten[0, 0, :, :].tolist():
                    #     print(['%.03f' % x for x in itm])
                    print('ori_factories atten_factories: ', torch.sum(old_states[:, self.policy.actor.obs_space.f_pos_dim_start, :, :]).tolist(),
                          torch.sum(f_atten[:, 0, :, :] * old_states[:, self.policy.actor.obs_space.f_pos_dim_start, :, :]).tolist())
                    print('#################################### u_tensor_one_hot unit_pos f_tensor_one_hot end #######################################')

                ed_loss = ed_loss_factor * self.mseLoss(self.policy.decoder(hidden), old_states / obs_normer)
                l1_regularization = l1_factor * sum([torch.norm(v, p=1) for v in self.policy.parameters()])
                l2_regularization = l2_factor * sum([torch.norm(v, p=2) for v in self.policy.parameters()])
                loss = f_loss + u_loss + v_loss + ed_loss + f_entropy_loss + u_entropy_loss + l1_regularization + l2_regularization
                # take gradient step
                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()
                tt_loss['v_loss'] += v_loss.mean().item()
                tt_loss['f_loss'] += f_loss.mean().item()
                tt_loss['u_loss'] += u_loss.mean().item()
                tt_loss['ed_loss'] += ed_loss.mean().item()
                tt += 1

        mean_loss = dict([(k, v / tt) for k, v in tt_loss.items()])
        log_writer.add_scalars('loss', mean_loss, online_agent_update_time)
        print(mean_loss, tt_loss)
        return self.policy.cpu().state_dict()
