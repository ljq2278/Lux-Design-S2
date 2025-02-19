"""
Implementation of RL agent. Note that luxai_s2 and stable_baselines3 are packages not available during the competition running (ATM)
"""
import multiprocessing
import copy
import os.path as osp

import gym
import numpy as np
import torch as th
import torch.nn as nn
from gym import spaces
from gym.wrappers import TimeLimit
from luxai_s2.state import ObservationStateDict, StatsStateDict
from luxai_s2.utils.heuristics.factory_placement import place_near_random_ice
from luxai_s2.wrappers import SB3Wrapper
import os
import matplotlib
import matplotlib.pyplot as plt
from lux.config import EnvConfig
from wrappers import MaActTransor, MaObsTransor, MaRwdTransor
from agent.BaseAgent import EarlyRuleAgent
from ppo.PPO import PPO
from ppo.UnitAgent import PPO_Offline_Agent
from ppo.UnitAgent import PPO_Online_Agent
from ppo.UnitBuffer import Buffer
import cv2

# matplotlib.use(backend='TkAgg')


class GlobalAgent(EarlyRuleAgent, PPO):
    def __init__(self, player: str, env_cfg: EnvConfig, unit_agent):
        EarlyRuleAgent.__init__(self, player, env_cfg)
        PPO.__init__(self, unit_agent)


env_id = "LuxAI_S2-v0"
print_interv = 10
actor_lr = 0.0004
critic_lr = 0.001
eps_clip = 0.2
K_epochs = 40
episode_num = 3000000
gamma = 0.98
sub_proc_count = 1
exp = 'paral_ppo2'
want_load_model = True
max_episode_length = 100
agent_debug = False
density_rwd = True


dim_info = [MaObsTransor.total_dims, MaActTransor.total_act_dims]  # obs and act dims
base_res_dir = os.environ['HOME'] + '/train_res/' + exp


def sub_run(replay_queue: multiprocessing.Queue, param_queue: multiprocessing.Queue, p_id):
    # unit_agent = unit_agent.value
    env = gym.make(env_id, verbose=0, collect_stats=True, MAX_FACTORIES=2)
    env_cfg = env.env_cfg
    env_cfg.max_episode_length = max_episode_length
    maActTransor = MaActTransor(env, env_cfg)
    maObsTransor = MaObsTransor(env, env_cfg, if_mask=True)
    maRwdTransor = MaRwdTransor(env, env_cfg, debug=False, density=density_rwd)
    agent_cont = 2
    unit_online_agent = PPO_Online_Agent(dim_info[0], dim_info[1])
    if want_load_model:
        new_params = param_queue.get()
        unit_online_agent.update(new_params)
    globalAgents = [GlobalAgent('player_' + str(i), env_cfg, unit_online_agent) for i in range(0, agent_cont)]
    globale_step = 0
    sum_rwd = 0
    for episode in range(0,1):
        np.random.seed()
        seed = np.random.randint(0, 100000000)
        raw_obs = env.reset(seed=seed)
        env.env_cfg.ROBOTS['HEAVY'].DIG_COST, env.env_cfg.ROBOTS['HEAVY'].MOVE_COST = 0, 0
        obs, norm_obs = maObsTransor.sg_to_ma(raw_obs['player_0'])
        done = {'player_0': False, 'player_1': False}
        imgs = []
        ################################ interact with the env for an episode ###################################
        while raw_obs['player_0']["real_env_steps"] < 0 or sum(done.values()) < len(done):
            if raw_obs['player_0']["real_env_steps"] < 0:
                raw_action = {}
                for g_agent in globalAgents:
                    raw_action[g_agent.player] = g_agent.early_setup(env.get_state().env_steps, raw_obs[g_agent.player])
                raw_next_obs, raw_reward, done, info = env.step(raw_action)
                raw_obs = raw_next_obs
            else:
                globale_step += 1
                ############################### get action and raw_action ###############################
                action = {}
                action_logprob = {}
                state_val = {}
                raw_action = {}
                for g_agent in globalAgents:
                    action[g_agent.player] = {}
                    action_logprob[g_agent.player] = {}
                    state_val[g_agent.player] = {}
                    for u_id, u_obs in norm_obs[g_agent.player].items():
                        a, b, c = unit_online_agent.policy.act([u_obs])
                        action[g_agent.player][u_id], action_logprob[g_agent.player][u_id], \
                            state_val[g_agent.player][u_id] = a[0], b[0], c[0][0]
                    raw_action[g_agent.player] = maActTransor.ma_to_sg(
                        action[g_agent.player], raw_obs[g_agent.player], g_agent.player)
                ############################### get action to env result ###############################
                raw_next_obs, raw_reward, done, info = env.step(raw_action)
                imgs += [env.render("rgb_array", width=640, height=640)]
                ############################### get next norm obs ######################################
                next_obs, norm_next_obs = maObsTransor.sg_to_ma(raw_next_obs['player_0'])
                ############################### get custom reward ######################################
                reward = {}
                for g_agent in globalAgents:
                    reward[g_agent.player] = maRwdTransor.sg_to_ma(
                        raw_reward[g_agent.player],
                        action[g_agent.player],
                        obs[g_agent.player],
                        next_obs[g_agent.player],
                        done[g_agent.player]
                    )
                    for u_id, u_info in norm_obs[g_agent.player].items():
                        if u_id not in norm_next_obs[g_agent.player].keys() or done[g_agent.player]:
                            d = True
                        else:
                            d = False
                        sum_rwd += reward[g_agent.player][u_id]
                ############################### print for debug #################################
                print('############################### print for debug #################################: ')
                print('norm_obs: ', norm_obs['player_0'])
                print('reward: ', reward['player_0'])
                print('action: ', action['player_0'])
                print('norm_next_obs: ', norm_next_obs['player_0'])
                ############################### prepare to the next step #################################
                raw_obs = raw_next_obs
                obs = next_obs
                norm_obs = norm_next_obs
        for img in imgs:
            cv2.imshow("Window", img)
            cv2.moveWindow('Window', 100, 100)
            cv2.waitKey(800)
            cv2.destroyAllWindows()

def offline_learn(replay_queue: multiprocessing.Queue, param_queue_list, pid):
    ppo_offline_agent = PPO_Offline_Agent(dim_info[0], dim_info[1], actor_lr, critic_lr, eps_clip, base_res_dir)
    if want_load_model:
        ppo_offline_agent.load()
        for param_queue in param_queue_list:
            param_queue.put(ppo_offline_agent.policy.state_dict())
    online_agent_update_time = 0
    train_data = []
    while True:
        if replay_queue.qsize() == len(param_queue_list):
            while replay_queue.qsize() != 0:
                data = replay_queue.get()
                train_data.append(data)
            new_params = ppo_offline_agent.update_and_get_new_param(train_data, K_epochs)
            online_agent_update_time += 1
            train_data.clear()
            for param_queue in param_queue_list:
                param_queue.put(new_params)



if __name__ == "__main__":
    replay_queue = multiprocessing.Queue()
    param_queue_list = [multiprocessing.Queue() for _ in range(0, sub_proc_count)]

    processes = []

    # Create some sub-processes that use the shared module
    for i in range(sub_proc_count):
        process = multiprocessing.Process(target=sub_run, args=(replay_queue, param_queue_list[i], i))
        process.start()
        processes.append(process)

    offline_learn_process = multiprocessing.Process(target=offline_learn, args=(replay_queue, param_queue_list, -1))
    offline_learn_process.start()
    processes.append(offline_learn_process)

    # Wait for the sub-processes to finish
    for process in processes:
        process.join()

    print('end .............')
    # Print the predictions
    # print(predictions)
