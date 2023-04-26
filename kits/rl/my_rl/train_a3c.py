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
from ac.AC import AC
from ac.UnitAgent import UnitAgent
from ac.UnitBuffer import Buffer


# matplotlib.use(backend='TkAgg')


class GlobalAgent(EarlyRuleAgent, AC):
    def __init__(self, player: str, env_cfg: EnvConfig, unit_agent):
        EarlyRuleAgent.__init__(self, player, env_cfg)
        AC.__init__(self, unit_agent)


def show(env):
    img = env.render("rgb_array", width=640, height=640)
    plt.imshow(img)
    plt.show()


env_id = "LuxAI_S2-v0"
buffer_capacity = 10
actor_lr = 0.0004
critic_lr = 0.0004
# total_lr = 0.001
episode_num = 3000000
gamma = 0.98
sub_proc_count = 10
exp = 'mask_obs'
want_load_model = True
max_episode_length = 20
agent_debug = False
density_rwd = True

dim_info = [MaObsTransor.total_dims, MaActTransor.total_act_dims]  # obs and act dims
base_res_dir = os.environ['HOME'] + '/train_res/' + exp
unit_buffer = Buffer(buffer_capacity, dim_info[0], dim_info[1], 'cpu')
unit_agent = UnitAgent(dim_info[0], dim_info[1], unit_buffer, base_res_dir, actor_lr, critic_lr, debug=agent_debug)
if want_load_model:
    unit_agent.load()


def sub_run(replay_queue: multiprocessing.Queue, action_queue: multiprocessing.Queue,
            obs_queue: multiprocessing.Queue, p_id):
    # unit_agent = unit_agent.value
    env = gym.make(env_id, verbose=0, collect_stats=True, MAX_FACTORIES=2)
    env_cfg = env.env_cfg
    env_cfg.max_episode_length = max_episode_length
    maActTransor = MaActTransor(env, env_cfg)
    maObsTransor = MaObsTransor(env, env_cfg, if_mask=True)
    maRwdTransor = MaRwdTransor(env, env_cfg, debug=False, density=density_rwd)
    agent_cont = 2
    globalAgents = [GlobalAgent('player_' + str(i), env_cfg, unit_agent) for i in range(0, agent_cont)]
    globale_step = 0
    for episode in range(episode_num):
        np.random.seed()
        seed = np.random.randint(0, 100000000)
        raw_obs = env.reset(seed=seed)
        obs, norm_obs = maObsTransor.sg_to_ma(raw_obs['player_0'])
        sum_rwd = 0
        done = {'player_0': False, 'player_1': False}
        ######################################################################### interact with the env for an episode

        while raw_obs['player_0']["real_env_steps"] < 0 or sum(done.values()) < len(done):
            if raw_obs['player_0']["real_env_steps"] < 0:
                raw_action = {}
                for g_agent in globalAgents:
                    raw_action[g_agent.player] = g_agent.early_setup(env.get_state().env_steps, raw_obs[g_agent.player])
                raw_next_obs, raw_reward, done, info = env.step(raw_action)
                raw_obs = raw_next_obs
            else:
                globale_step += 1
                action = {}
                raw_action = {}
                for g_agent in globalAgents:
                    action[g_agent.player] = {}
                    for u_id, u_obs in norm_obs[g_agent.player].items():
                        obs_queue.put(u_obs)
                        action[g_agent.player][u_id] = action_queue.get()
                    # action[g_agent.player] = g_agent.get_action(norm_obs[g_agent.player])
                    raw_action[g_agent.player] = maActTransor.ma_to_sg(
                        action[g_agent.player], raw_obs[g_agent.player], g_agent.player)

                raw_next_obs, raw_reward, done, info = env.step(raw_action)
                next_obs, norm_next_obs = maObsTransor.sg_to_ma(raw_next_obs['player_0'])
                reward = {}
                for g_agent in globalAgents:
                    reward[g_agent.player] = maRwdTransor.sg_to_ma(
                        raw_reward[g_agent.player],
                        action[g_agent.player],
                        obs[g_agent.player],
                        next_obs[g_agent.player],
                        done[g_agent.player]
                        # raw_obs['player_0']['board']['ice'],
                        # raw_obs['player_0']['board']['ore']
                    )
                    for u_id, u_info in norm_obs[g_agent.player].items():
                        if u_id not in norm_next_obs[g_agent.player].keys():
                            norm_next_o = norm_obs[g_agent.player][u_id]
                            d = True
                        else:
                            norm_next_o = norm_next_obs[g_agent.player][u_id]
                            d = False
                        replay_queue.put([norm_obs[g_agent.player][u_id], action[g_agent.player][u_id],
                                          reward[g_agent.player][u_id], norm_next_o, d])

                    sum_rwd += sum([v for v in reward[g_agent.player].values()])

                raw_obs = raw_next_obs
                obs = next_obs
                norm_obs = norm_next_obs
                # episode finishes
                if (globale_step + 1) % 50 == 0:  # print info every 100 g_step
                    message = f'episode {episode + 1}, '
                    message += f'sum reward: {sum_rwd}'
                    print(message)
                    print(maRwdTransor.reward_collect)
                    for k, v in maRwdTransor.reward_collect.items():
                        maRwdTransor.reward_collect[k] = 0


def sub_obs_action_learn(replay_queue: multiprocessing.Queue, action_queue_list, obs_queue_list, p_id):
    cont = 0
    while True:
        for p in range(len(action_queue_list)):
            obs = obs_queue_list[p].get()
            act = unit_agent.get_action(obs)
            action_queue_list[p].put(act)
        while replay_queue.qsize() != 0:
            replay = replay_queue.get()
            o, a, r, next_o, d = replay
            unit_agent.unit_buffer.add(o, a, r, next_o, d)
            if unit_agent.unit_buffer.size == buffer_capacity:
                cont += 1
                unit_agent.learn(gamma)
                if (cont + 1) % (10000 // buffer_capacity) == 0:
                    print('save model, learning time ', cont)
                    unit_agent.save()


if __name__ == "__main__":
    replay_queue = multiprocessing.Queue()
    action_queues = [multiprocessing.Queue() for _ in range(0, sub_proc_count)]
    obs_queues = [multiprocessing.Queue() for _ in range(0, sub_proc_count)]

    processes = []

    # Create some sub-processes that use the shared module
    for i in range(sub_proc_count):
        process = multiprocessing.Process(target=sub_run, args=(replay_queue, action_queues[i], obs_queues[i], i))
        process.start()
        processes.append(process)

    sub_obs_action_learn_process = multiprocessing.Process(target=sub_obs_action_learn,
                                                           args=(replay_queue, action_queues, obs_queues, -1))
    sub_obs_action_learn_process.start()
    processes.append(sub_obs_action_learn_process)

    # Wait for the sub-processes to finish
    for process in processes:
        process.join()

    print('end .............')
    # Print the predictions
    # print(predictions)
