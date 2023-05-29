"""
Implementation of RL agent. Note that luxai_s2 and stable_baselines3 are packages not available during the competition running (ATM)
"""
import multiprocessing
import copy
import os.path as osp

import gym
import numpy as np
import os
import luxai_s2
from lux.config import EnvConfig
from wrappers.act_transfer import ActTransfer
from wrappers.rwd_transfer import RwdTransfer
from wrappers.obs_transfer import ObsTransfer
from agent.BaseAgent import EarlyRuleAgent

from wrappers.obs_space_conv import ObsSpace
from wrappers.act_space_conv import ActSpaceFactory, ActSpaceUnit
from ppo_conv.Buffer import Buffer
from ppo_conv.CentralAgent import CentralOnlineAgent, CentralOfflineAgent
import matplotlib.pyplot as plt
from luxai_s2.map.position import Position
import torch


class GlobalAgent(EarlyRuleAgent):
    def __init__(self, player: str, env_cfg: EnvConfig, central_agent):
        EarlyRuleAgent.__init__(self, player, env_cfg)
        self.central_agent = central_agent


torch.set_printoptions(threshold=10000000)

env_id = "LuxAI_S2-v0"
debug = True
debug_param = False
debug_decoder = True
gumbel_softmax_tau = 1
gamma = 0.98
exp = 'paral_ppo_share'
max_episode_length = 100
agent_debug = False
density_rwd = False
epochs = 1
map_size = 24
os.environ['HOME'] = 'D:'
want_load_model = True
dim_info = [ObsSpace(None).total_dims, ActSpaceFactory().f_dims, ActSpaceUnit().u_dims]  # obs and act dims
base_res_dir = os.environ['HOME'] + '/train_res/' + exp

if __name__ == "__main__":
    env = gym.make(env_id, verbose=0, collect_stats=True, MAX_FACTORIES=3)
    env_cfg = env.env_cfg
    env_cfg.map_size = map_size
    env_cfg.max_episode_length = max_episode_length
    actTransfer = ActTransfer(env, env_cfg)
    obsTransfer = ObsTransfer(env, env_cfg)
    rwdTransfer = RwdTransfer(env, env_cfg, debug=False, density=density_rwd)
    agent_cont = 2
    online_agent = CentralOnlineAgent(dim_info[0], dim_info[1], dim_info[2], env_cfg, gumbel_softmax_tau=gumbel_softmax_tau, save_dir=base_res_dir, is_cuda=False)
    if want_load_model:
        new_params = online_agent.load()
    globalAgents = [GlobalAgent('player_' + str(i), env_cfg, online_agent) for i in range(0, agent_cont)]
    globale_step = 0
    sum_rwd = 0
    survive_step = 0
    buffer = Buffer()
    tmp_buffer = {}  # record every datas
    for episode in range(0, epochs):
        np.random.seed()
        seed = np.random.randint(0, 100000000)
        seed = 88
        raw_obs = env.reset(seed=seed)
        done = {'player_0': False, 'player_1': False}
        ################################ interact with the env for an episode ###################################
        while raw_obs['player_0']["real_env_steps"] < 0 or sum(done.values()) < len(done):
            globale_step += 1
            raw_action = {'player_0': {}, 'player_1': {}}
            if raw_obs['player_0']["real_env_steps"] < 0:
                for g_agent in globalAgents:
                    raw_action[g_agent.player] = g_agent.early_setup(env.get_state().env_steps, raw_obs[g_agent.player])
                raw_next_obs, raw_reward, done, info = env.step(raw_action)
                raw_obs = raw_next_obs
            elif raw_obs['player_0']["real_env_steps"] == 0:
                for p_id, fp_info in env.state.factories.items():
                    for f_id in fp_info.keys():
                        raw_action[p_id][f_id] = 1
                        if p_id == 'player_1':
                            # set factories to have 1000 water to check the ore dig ability
                            env.state.factories[p_id][f_id].cargo.water = 200
                            # env.state.factories[p_id][f_id].cargo.metal = 200
                            # env.state.factories[p_id][f_id].power = 300000
                        else:
                            env.state.factories[p_id][f_id].cargo.water = 40
                            env.state.factories[p_id][f_id].cargo.metal += 2000  # int(1000 * np.random.random())
                            env.state.factories[p_id][f_id].power = 300000
                print(raw_obs['player_0']["real_env_steps"], raw_action['player_0'])
                raw_next_obs, raw_reward, done, info = env.step(raw_action)
                print(raw_obs['player_0']["real_env_steps"], env.state.stats['player_0'])
                if debug_param:
                    for name, param in online_agent.policy.actor.named_parameters():
                        if param.requires_grad:
                            print(name, param.data)
                    for name, param in online_agent.policy.critic.named_parameters():
                        if param.requires_grad:
                            print(name, param.data)
                obs, obs_stat = obsTransfer.raw_to_wrap(raw_obs['player_0'], env.state, max_episode_length - raw_obs['player_0']["real_env_steps"])
                last_stats = copy.deepcopy(env.state)
                raw_obs = raw_next_obs
            else:
                ############################### get action and raw_action factory ###############################
                f_action, u_action = {}, {}
                f_action_logprob, u_action_logprob = {}, {}
                state_val = {}
                for g_agent in [globalAgents[0]]:
                    f_action[g_agent.player], u_action[g_agent.player] = {}, {}
                    f_action_logprob[g_agent.player], u_action_logprob[g_agent.player] = {}, {}
                    state_val[g_agent.player], f_action[g_agent.player], f_action_logprob[g_agent.player], u_action[g_agent.player], u_action_logprob[g_agent.player], h \
                        = online_agent.policy.act(np.array([obs[g_agent.player]]), np.array([obs_stat[g_agent.player]]), device='cpu')
                    if debug_decoder:
                        print('####################################### hidden: start ###############################################')
                        # for i in range(0, h.size()[1]):
                        #     print(h[0, i, :, :])
                        print('######################################## hidden: end ###############################################')
                        print(np.array(u_action[g_agent.player]))
                        # print(online_agent.policy.decoder(h)[0, online_agent.policy.critic.obs_space.b_ice_dim_start, :, :])
                        # print(torch.argwhere(online_agent.policy.decoder(h)[0, online_agent.policy.critic.obs_space.b_ice_dim_start, :, :] > 0.0))
                        print(torch.argwhere(torch.Tensor(obs[g_agent.player][online_agent.policy.critic.obs_space.b_ice_dim_start, :, :]) > 0.5))
                        # print(online_agent.policy.decoder(h)[0, online_agent.policy.critic.obs_space.b_ice_dim, :, :])
                    state_val[g_agent.player], f_action[g_agent.player], f_action_logprob[g_agent.player], u_action[g_agent.player], u_action_logprob[g_agent.player] \
                        = state_val[g_agent.player][0][0], f_action[g_agent.player][0], f_action_logprob[g_agent.player][0], u_action[g_agent.player][0], u_action_logprob[g_agent.player][0]
                    raw_action[g_agent.player] = actTransfer.wrap_to_raw(
                        f_action[g_agent.player], u_action[g_agent.player], env.state.factories[g_agent.player], env.state.units[g_agent.player])
                ############################### get action to env result ###############################
                raw_next_obs, raw_reward, done, info = env.step(raw_action)
                img = env.render("rgb_array", width=640, height=640)
                plt.imsave('./imgs/' + str(raw_obs['player_0']["real_env_steps"]) + '.png', img)
                ############################### get next obs factory ######################################
                next_obs, next_obs_stat = obsTransfer.raw_to_wrap(raw_next_obs['player_0'], env.state, max_episode_length - raw_obs['player_0']["real_env_steps"])
                ############################### get custom reward factory ######################################
                reward = {}
                step_reward = {}
                for g_agent in [globalAgents[0]]:
                    reward[g_agent.player], step_reward[g_agent.player] = rwdTransfer.raw_to_wrap(
                        raw_reward[g_agent.player],
                        done[g_agent.player],
                        last_stats.stats[g_agent.player],
                        env.state.stats[g_agent.player],
                        last_stats.factories[g_agent.player],
                        env.state.factories[g_agent.player],
                        last_stats.units[g_agent.player],
                        env.state.units[g_agent.player],
                        obs[g_agent.player],
                        next_obs[g_agent.player]
                    )
                    sum_rwd += reward[g_agent.player]
                if debug:
                    print('################################################# ', raw_obs['player_0']["real_env_steps"],
                          ' start ###########################################################################')
                    print('state map: ', np.max(u_action['player_0']), np.min(u_action['player_0']))
                    # for itm in u_action['player_0'] * obs['player_0'][obsTransfer.obs_space.u_pos_dim_start, :, :]:
                    #     print(['%i' % x for x in itm])
                    # for itm in obs['player_0'][obsTransfer.obs_space.b_rub_dim_start, :, :]:
                    #     print(['%03i' % x for x in itm])
                    print('state: ', last_stats.stats['player_0'])
                    print('state_val: ', state_val['player_0'])
                    print('obs_stat: ', obs_stat['player_0'])
                    print('action: ', raw_action['player_0'])
                    # print('u_action_logprob: ', u_action_logprob['player_0'])
                    print('reward: ', step_reward['player_0'])
                    print('next_state: ', env.state.stats['player_0'])
                    print('################################################# ', raw_obs['player_0']["real_env_steps"],
                          ' end #############################################################################')
                ############################### prepare to the next step #################################
                raw_obs = raw_next_obs
                print('max and min: ', np.max(obs['player_0'] / (obsTransfer.obs_space.normer.reshape([-1, 1, 1])) - next_obs['player_0'] / (obsTransfer.obs_space.normer.reshape([-1, 1, 1]))),
                      np.min(obs['player_0'] / (obsTransfer.obs_space.normer.reshape([-1, 1, 1])) - next_obs['player_0'] / (obsTransfer.obs_space.normer.reshape([-1, 1, 1]))))
                obs = next_obs
                obs_stat = next_obs_stat
                last_stats = copy.deepcopy(env.state)
        ############################### episode data record  #################################
        survive_step += raw_obs["player_0"]["real_env_steps"]
        # print(seed, raw_obs["player_0"]["real_env_steps"])
    print('############################################### rewards ######################################################')
    print('avg rewards: ', sum_rwd / epochs)
    print('avg survive: ', survive_step / epochs)
    for k, v in rwdTransfer.reward_collect.items():
        print(k, v / epochs)
