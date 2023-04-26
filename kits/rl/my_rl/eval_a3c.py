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


class GlobalAgent(EarlyRuleAgent, AC):
    def __init__(self, player: str, env_cfg: EnvConfig, unit_agent):
        EarlyRuleAgent.__init__(self, player, env_cfg)
        AC.__init__(self, unit_agent)


env_id = "LuxAI_S2-v0"
buffer_capacity = 10
actor_lr = 0.001
critic_lr = 0.001
episode_num = 3000000
gamma = 0.9
sub_proc_count = 10
exp = 'mask_obs'
max_episode_length = 20
agent_debug = False


def eval(env_id):
    env = gym.make(env_id, verbose=0, collect_stats=True, MAX_FACTORIES=2)
    env_cfg = env.env_cfg
    env_cfg.max_episode_length = max_episode_length
    maActTransor = MaActTransor(env, env_cfg)
    maObsTransor = MaObsTransor(env, env_cfg, if_mask=True)
    maRwdTransor = MaRwdTransor(env, env_cfg, debug=True, density=True)
    agent_cont = 2
    dim_info = [maObsTransor.total_dims, maActTransor.total_act_dims]  # obs and act dims
    base_res_dir = os.environ['HOME'] + '/train_res/' + exp
    unit_buffer = Buffer(buffer_capacity, dim_info[0], dim_info[1], 'cpu')
    unit_agent = UnitAgent(dim_info[0], dim_info[1],  unit_buffer, base_res_dir, actor_lr, critic_lr)

    unit_agent.load()

    globalAgents = [GlobalAgent('player_' + str(i), env_cfg, unit_agent) for i in range(0, agent_cont)]
    globale_step = 0
    for episode in range(episode_num):
        np.random.seed()
        seed = np.random.randint(0, 10000000)
        raw_obs = env.reset(seed=seed)
        obs, norm_obs = maObsTransor.sg_to_ma(raw_obs['player_0'])
        sum_rwd = 0
        step = 0
        # early_steps = -raw_obs['player_0']["real_env_steps"]
        done = {'player_0': False, 'player_1': False}
        imgs = []
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
                # print(env.get_state().stats['player_0'])
                imgs += [env.render("rgb_array", width=640, height=640)]
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

                raw_obs = raw_next_obs
                obs = next_obs
                norm_obs = norm_next_obs

            # show(env)
        for img in imgs:
            cv2.imshow("Window", img)
            cv2.moveWindow('Window', 100, 100)
            cv2.waitKey(800)
            cv2.destroyAllWindows()
            # img = Image.fromarray(img)
            # img.show()
            # time.sleep(0.5)
            # img.close()

        # episode finishes
        if (episode + 1) % 10 == 0:  # print info every 100 episodes
            message = f'episode {episode + 1}, '
            message += f'sum reward: {sum_rwd}'
            print(message, globale_step)
            units_info = [(v.power, v.cargo) for uid, v in env.get_state().units['player_0'].items()]
            state_info = env.get_state().stats['player_0']
            print(state_info)
            for unit_info in units_info:
                print(unit_info)


if __name__ == "__main__":
    eval(env_id)
