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


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    # parser.add_argument('env_name', type=str, default='simple_adversary_v2', help='name of the env',
    #                     choices=['simple_adversary_v2', 'simple_spread_v2', 'simple_tag_v2'])
    parser.add_argument('--episode_num', type=int, default=3000000,
                        help='total episode num during training procedure')
    parser.add_argument('--gamma', type=float, default=0.98, help='discount factor')
    parser.add_argument('--buffer_capacity', type=int, default=int(1e3), help='capacity of replay buffer')
    parser.add_argument('--actor_lr', type=float, default=0.0005, help='learning rate of actor')
    parser.add_argument('--critic_lr', type=float, default=0.0005, help='learning rate of critic')
    args = parser.parse_args()
    return args


def show(env):
    img = env.render("rgb_array", width=640, height=640)
    plt.imshow(img)
    plt.show()

exp = 'B'
max_episode_length = 20

def train(args, env_id):
    env = gym.make(env_id, verbose=0, collect_stats=True, MAX_FACTORIES=3)
    env_cfg = env.env_cfg
    env_cfg.max_episode_length = max_episode_length
    maActTransor = MaActTransor(env, env_cfg)
    maObsTransor = MaObsTransor(env, env_cfg)
    maRwdTransor = MaRwdTransor(env, env_cfg, debug=True)

    dim_info = [maObsTransor.total_dims, maActTransor.total_act_dims]  # obs and act dims
    base_res_dir = os.environ['HOME'] + '/train_res/' + exp
    agent_cont = 2
    unit_buffer = Buffer(args.buffer_capacity, dim_info[0], dim_info[1], 'cpu')
    unit_agent = UnitAgent(dim_info[0], dim_info[1], args.actor_lr, args.critic_lr, unit_buffer, base_res_dir)
    # unit_agent.load()
    globalAgents = [GlobalAgent('player_' + str(i), env_cfg, unit_agent) for i in range(0, agent_cont)]
    globale_step = 0
    for episode in range(args.episode_num):
        np.random.seed()
        seed = np.random.randint(0, 100000000)
        raw_obs = env.reset(seed=seed)
        obs, norm_obs = maObsTransor.sg_to_ma(raw_obs['player_0'])
        sum_rwd = 0
        step = 0
        # early_steps = -raw_obs['player_0']["real_env_steps"]
        done = {'player_0': False, 'player_1': False}

        ######################################################################### interact with the env for an episode

        while raw_obs['player_0']["real_env_steps"] < 0 or sum(done.values()) < len(done):
            # if (episode + 1) % 1000 == 0:
            #     show(env)
            if raw_obs['player_0']["real_env_steps"] < 0:
                raw_action = {}
                for g_agent in globalAgents:
                    raw_action[g_agent.player] = g_agent.early_setup(step, raw_obs[g_agent.player])
                raw_next_obs, raw_reward, done, info = env.step(raw_action)
                raw_obs = raw_next_obs
                step += 1
            else:
                globale_step += 1
                action = {}

                for g_agent in globalAgents:
                    action[g_agent.player] = {}
                    for u_id, u_obs in norm_obs[g_agent.player].items():
                        action[g_agent.player][u_id] = unit_agent.get_action(norm_obs[g_agent.player][u_id])

                raw_action = {}
                for g_agent in globalAgents:
                    raw_action[g_agent.player] = maActTransor.ma_to_sg(action[g_agent.player], raw_obs[g_agent.player],
                                                                       g_agent.player)
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

                    for u_id, u_obs in norm_obs[g_agent.player].items():
                        if u_id not in norm_next_obs[g_agent.player].keys():
                            norm_next_o = norm_obs[g_agent.player][u_id]
                            d = True
                        else:
                            norm_next_o = norm_next_obs[g_agent.player][u_id]
                            d = False
                        unit_agent.unit_buffer.add(norm_obs[g_agent.player][u_id],
                                                   action[g_agent.player][u_id],
                                                   reward[g_agent.player][u_id], norm_next_o, d)
                    sum_rwd += sum([v for v in reward[g_agent.player].values()])
                    # learn only one time ! that is what policy gradient ask for
                    if g_agent.unit_agent.unit_buffer.size == args.buffer_capacity:
                        unit_agent.learn(args.gamma)

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
            unit_agent.save()


def main(args):
    print("Training with args", args)
    # if args.seed is not None:
    #     set_random_seed(args.seed)
    env_id = "LuxAI_S2-v0"

    train(args, env_id)


if __name__ == "__main__":
    # python ../examples/sb3.py -l logs/exp_1 -s 42 -n 1
    main(parse_args())
