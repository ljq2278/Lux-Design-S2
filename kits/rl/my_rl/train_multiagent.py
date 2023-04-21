"""
Implementation of RL agent. Note that luxai_s2 and stable_baselines3 are packages not available during the competition running (ATM)
"""

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

from wrappers import MaActTransor, MaObsTransor, MaRwdTransor
from GlobalAgent import GlobalAgent

# matplotlib.use(backend='TkAgg')

exp = 'B'


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    # parser.add_argument('env_name', type=str, default='simple_adversary_v2', help='name of the env',
    #                     choices=['simple_adversary_v2', 'simple_spread_v2', 'simple_tag_v2'])
    parser.add_argument('--episode_num', type=int, default=3000000,
                        help='total episode num during training procedure')
    # parser.add_argument('--episode_length', type=int, default=25, help='steps per episode')
    parser.add_argument('--learn_interval', type=int, default=1000,
                        help='steps interval between learning time')
    parser.add_argument('--random_search', type=float, default=0.1,
                        help='random search probability')
    parser.add_argument('--tau', type=float, default=0.05, help='soft update parameter')
    parser.add_argument('--gamma', type=float, default=0.98, help='discount factor')
    parser.add_argument('--buffer_capacity', type=int, default=int(1e6), help='capacity of replay buffer')
    parser.add_argument('--batch_size', type=int, default=1000, help='batch-size of replay buffer')
    parser.add_argument('--actor_lr', type=float, default=0.01, help='learning rate of actor')
    parser.add_argument('--critic_lr', type=float, default=0.01, help='learning rate of critic')
    args = parser.parse_args()
    return args


def show(env):
    img = env.render("rgb_array", width=640, height=640)
    plt.imshow(img)
    plt.show()


def train(args, env_id):
    env = gym.make(env_id, verbose=0, collect_stats=True, MAX_FACTORIES=3)
    env_cfg = env.env_cfg
    maActTransor = MaActTransor(env_cfg)
    maObsTransor = MaObsTransor(env_cfg)
    maRwdTransor = MaRwdTransor(env, env_cfg, debug=True)
    dim_info = [maObsTransor.total_dims, maActTransor.total_act_dims]  # obs and act dims
    base_res_dir = os.environ['HOME']+'/train_res/' + exp + '/'
    agent_cont = 2
    globalAgents = [GlobalAgent(dim_info, args.buffer_capacity, args.batch_size, args.actor_lr, args.critic_lr,
                                base_res_dir + str(i), 'player_' + str(i), env_cfg) for i in
                    range(0, agent_cont)]
    for i, g_agent in enumerate(globalAgents):
        g_agent.load(base_res_dir + '0/model.pt')
        print('loading model .................')
    globale_step = 0
    for episode in range(args.episode_num):
        raw_obs = env.reset()
        obs, norm_obs = maObsTransor.sg_to_ma(raw_obs['player_0'])
        sum_rwd = 0
        step = 0
        # early_steps = -raw_obs['player_0']["real_env_steps"]
        done = {'player_0': False, 'player_1': False}

        ######################################################################### interact with the env for an episode

        while raw_obs['player_0']["real_env_steps"] < 0 or sum(done.values()) < len(done):
            # if (episode + 1) % 1000 == 0:
            #     show(env)
            globale_step += 1
            if raw_obs['player_0']["real_env_steps"] < 0:
                raw_action = {}
                for g_agent in globalAgents:
                    raw_action[g_agent.player] = g_agent.early_setup(step, raw_obs[g_agent.player])
                raw_next_obs, raw_reward, done, info = env.step(raw_action)
                raw_obs = raw_next_obs
                step += 1
            else:
                action = {}
                if np.random.random() < args.random_search:
                    for g_agent in globalAgents:
                        action[g_agent.player] = {unit_id: maActTransor.action_space.sample() for unit_id, _ in
                                                  norm_obs[g_agent.player].items()}
                else:
                    for g_agent in globalAgents:
                        action[g_agent.player] = g_agent.select_action(
                            norm_obs[g_agent.player])  # it should be a stand obs and  for maddpg

                raw_action = {}
                for g_agent in globalAgents:
                    raw_action[g_agent.player] = maActTransor.ma_to_sg(action[g_agent.player], raw_obs[g_agent.player],
                                                                       g_agent.player)
                raw_next_obs, raw_reward, done, info = env.step(raw_action)
                next_obs, norm_next_obs = maObsTransor.sg_to_ma(raw_next_obs['player_0'])
                reward = {}
                for g_agent in globalAgents:
                    reward[g_agent.player] = maRwdTransor.sg_to_ma(
                        raw_reward[g_agent.player], action[g_agent.player],
                        obs[g_agent.player],
                        next_obs[g_agent.player],
                        raw_obs['player_0']['board']['ice'],
                        raw_obs['player_0']['board']['ore']
                    )
                    g_agent.add(norm_obs[g_agent.player], action[g_agent.player], reward[g_agent.player],
                                norm_next_obs[g_agent.player],
                                done[g_agent.player])  # it should be a stand reward and action for maddpg
                    sum_rwd += sum([v for v in reward[g_agent.player].values()])
                    if globale_step > args.buffer_capacity//4 and globale_step % args.learn_interval == 0:  # learn every few steps
                        g_agent.learn(args.batch_size, args.gamma)
                        g_agent.update_target(args.tau)
                raw_obs = raw_next_obs
                obs = next_obs
                norm_obs = norm_next_obs
            # show(env)
        # episode finishes
        if (episode + 1) % 10 == 0:  # print info every 100 episodes
            message = f'episode {episode + 1}, '
            message += f'sum reward: {sum_rwd}'
            print(message)
            units_info = [(v.power, v.cargo) for uid, v in env.get_state().units['player_0'].items()]
            state_info = env.get_state().stats['player_0']
            print(state_info)
            for unit_info in units_info:
                print(unit_info)
        if (episode + 1) % 100 == 0:  # print info every 100 episodes
            for g_agent in globalAgents:
                g_agent.save()  # save model
                break


def main(args):
    print("Training with args", args)
    # if args.seed is not None:
    #     set_random_seed(args.seed)
    env_id = "LuxAI_S2-v0"

    train(args, env_id)


if __name__ == "__main__":
    # python ../examples/sb3.py -l logs/exp_1 -s 42 -n 1
    main(parse_args())
