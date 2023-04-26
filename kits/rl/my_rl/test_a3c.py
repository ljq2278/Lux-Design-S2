"""
Implementation of RL agent. Note that luxai_s2 and stable_baselines3 are packages not available during the competition running (ATM)
"""

import copy
import os.path as osp
from PIL import Image
import cv2
import time
import gym
import numpy as np
import torch as th
import torch.nn as nn
import luxai_s2

import os
import matplotlib
import matplotlib.pyplot as plt
from lux.config import EnvConfig
from wrappers import MaActTransor, MaObsTransor, MaRwdTransor
from agent.BaseAgent import EarlyRuleAgent
from ac.AC import AC
from ac.UnitAgent import UnitAgent
from ac.UnitBuffer import Buffer

matplotlib.use(backend='TkAgg')

exp = 'mask_obs'
human_control = 'True'


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
    parser.add_argument('--actor_lr', type=float, default=0.01, help='learning rate of actor')
    parser.add_argument('--critic_lr', type=float, default=0.01, help='learning rate of critic')
    args = parser.parse_args()
    return args


def test(args, env_id):
    env = gym.make(env_id, verbose=0, collect_stats=True, MAX_FACTORIES=3)
    env_cfg = env.env_cfg
    env_cfg.max_episode_length = 20
    maActTransor = MaActTransor(env, env_cfg)
    maObsTransor = MaObsTransor(env, env_cfg, if_mask=True)
    maRwdTransor = MaRwdTransor(env, env_cfg, debug=True, density=True)
    agent_cont = 2
    dim_info = [maObsTransor.total_dims, maActTransor.total_act_dims]  # obs and act dims
    base_res_dir = os.environ['HOME'] + '/train_res/' + exp
    unit_buffer = Buffer(args.buffer_capacity, dim_info[0], dim_info[1], 'cpu')
    unit_agent = UnitAgent(dim_info[0], dim_info[1], args.actor_lr, args.critic_lr, unit_buffer, base_res_dir)

    unit_agent.load()

    globalAgents = [GlobalAgent('player_' + str(i), env_cfg, unit_agent) for i in range(0, agent_cont)]
    globale_step = 0
    for episode in range(args.episode_num):
        np.random.seed()
        seed = np.random.randint(0, 10000000)
        raw_obs = env.reset(seed=seed)
        obs, norm_obs = maObsTransor.sg_to_ma(raw_obs['player_0'])
        sum_rwd = 0
        step = 0
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
                        if not human_control:
                            action[g_agent.player][u_id] = unit_agent.get_action(norm_obs[g_agent.player][u_id])
                        else:
                            action[g_agent.player][u_id] = int(
                                input('_'.join([g_agent.player, u_id]) + ' give an action: '))
                print('act ... ')
                raw_action = {}
                for g_agent in globalAgents:
                    raw_action[g_agent.player] = maActTransor.ma_to_sg(action[g_agent.player], raw_obs[g_agent.player],
                                                                       g_agent.player)
                raw_next_obs, raw_reward, done, info = env.step(raw_action)
                img = env.render("rgb_array", width=640, height=640)
                img = Image.fromarray(img)
                img.show()
                # print(env.get_state().stats['player_0'])
                next_obs, norm_next_obs = maObsTransor.sg_to_ma(raw_next_obs['player_0'])
                reward = {}
                for g_agent in globalAgents:
                    reward[g_agent.player] = maRwdTransor.sg_to_ma(
                        raw_reward[g_agent.player], action[g_agent.player],
                        obs[g_agent.player],
                        next_obs[g_agent.player],
                        done[g_agent.player]
                    )
                    sum_rwd += sum([v for v in reward[g_agent.player].values()])
                print('reward:', reward, maRwdTransor.reward_collect)
                for k, v in maRwdTransor.reward_collect.items():
                    maRwdTransor.reward_collect[k] = 0
                raw_obs = raw_next_obs
                obs = next_obs
                norm_obs = norm_next_obs


def main(args):
    print("Training with args", args)
    # if args.seed is not None:
    #     set_random_seed(args.seed)
    env_id = "LuxAI_S2-v0"

    test(args, env_id)


if __name__ == "__main__":
    # python ../examples/sb3.py -l logs/exp_1 -s 42 -n 1
    main(parse_args())
