from wrappers.ma_act_transor import MaActTransor
from wrappers.ma_obs_transor import MaObsTransor
from wrappers.ma_rwd_transor import MaRwdTransor
import numpy as np

import gym
from gym import spaces

import copy
import os.path as osp

import gym
import torch as th
import torch.nn as nn
from gym import spaces
from gym.wrappers import TimeLimit
from luxai_s2.state import ObservationStateDict, StatsStateDict
from luxai_s2.utils.heuristics.factory_placement import place_near_random_ice
from luxai_s2.wrappers import SB3Wrapper
from lux.config import EnvConfig

import json
from collections import defaultdict

nested_dict = lambda: defaultdict(nested_dict)



board_ice = np.zeros([48, 48], float)
board_ore = np.zeros([48, 48], float)
board_rub = np.zeros([48, 48], float)

ice_locs = [[3, 4], [25, 8], [12, 1]]
ore_locs = [[46, 2], [37, 23]]

rub_loc_values = [
    ([12, 13], 24),
    ([13, 13], 22),
    ([13, 14], 20),
]

for x,y in ice_locs:
    board_ice[x,y] = 1
for x,y in ore_locs:
    board_ore[x,y] = 1
for (x,y),v in rub_loc_values:
    board_rub[x,y] = v
# bots_loc = {
#     'player_0': [[24, 10], [1, 40]],
#     'player_1': [[4, 4], [20, 20], [14, 30]]
# }


player_0 = nested_dict()
player_0['unit_1']['pos'] = [24, 10]
player_0['unit_1']['power'] = 100
player_0['unit_1']['cargo']['ice'] = 10
player_0['unit_1']['cargo']['ore'] = 5

player_0['unit_2']['pos'] = [1, 40]
player_0['unit_2']['power'] = 120
player_0['unit_2']['cargo']['ice'] = 15
player_0['unit_2']['cargo']['ore'] = 8

player_1 = nested_dict()
player_1['unit_1']['pos'] = [24, 12]
player_1['unit_1']['power'] = 10
player_1['unit_1']['cargo']['ice'] = 100
player_1['unit_1']['cargo']['ore'] = 50

player_1['unit_2']['pos'] = [29, 30]
player_1['unit_2']['power'] = 12
player_1['unit_2']['cargo']['ice'] = 150
player_1['unit_2']['cargo']['ore'] = 80

player_0_factories = nested_dict()
player_0_factories['factory_0']['pos'] = [2, 2]
player_0_factories['factory_1']['pos'] = [4, 4]

player_1_factories = nested_dict()
player_1_factories['factory_0']['pos'] = [42, 2]
player_1_factories['factory_1']['pos'] = [42, 4]


raw_obs = json.load(open('../data_example/raw_obs.json','r',encoding='utf-8'))
raw_obs['player_0']['board']['ice'] = board_ice
raw_obs['player_0']['board']['ore'] = board_ore
raw_obs['player_0']['board']['rubble'] = board_rub
raw_obs['player_0']['units']['player_0'] = player_0
raw_obs['player_0']['units']['player_1'] = player_1
raw_obs['player_0']['factories']['player_0'] = player_0_factories
raw_obs['player_0']['factories']['player_1'] = player_1_factories
raw_obs['player_1'] = raw_obs['player_0']


env_id = "LuxAI_S2-v0"
env = gym.make(env_id, verbose=0, collect_stats=True, MAX_FACTORIES=2)
env_cfg = env.env_cfg
maObsTransor = MaObsTransor(env_cfg)
obs = maObsTransor.sg_to_ma(raw_obs['player_1'])
tt = 1

