from typing import Any, Dict

import gym
import numpy as np
import numpy.typing as npt
from gym import spaces
# import matplotlib.pyplot as plt
from lux.config import EnvConfig
from wrappers.obs_space_levels import ObsSpaceUnit, ObsSpaceFactory
from wrappers.act_space_levels import ActSpaceUnit, ActSpaceFactory, ActSpaceFactoryDemand


class RwdTransfer:

    def __init__(self, env, env_cfg: EnvConfig, debug=False, density=True) -> None:
        self.env_cfg = env_cfg
        self.env = env
        self.debug = debug
        self.density = density
        self.reward_collect = {
            'ice_generation': 0,
            'ore_generation': 0,
            'power_generation': 0,
            'water_generation': 0,
            'metal_generation': 0,
            'ice_to_factory': 0,
            'ore_to_factory': 0,
            # 'ice_transfer': 0,
            # 'ore_transfer': 0,
            'destroyed_rubble': 0,
            # 'leave the way rubble': 0,
            # 'on the way home with target': 0,
            # 'leave the way home with target': 0,
            # 'collision happen': 0,
            # 'transfer target': 0,
            # 'transfer ice': 0,
            # 'transfer ore': 0,
            # 'dig out target rubble': 0,
            # 'low power charged': 0,
            'step': 0,
            'ori_reward': 0,
            # 'left_step_punish': 0
        }
        return

    def raw_to_wrap(self, ori_reward, done, last_stats, stats, last_factories, factories, last_units, units):
        step_reward = {
            'step': 1.0,
            'ice_to_factory': sum([max((f_info.cargo.ice - lf_info.cargo.ice), 0) for ((f_id, f_info), (lf_id, lf_info)) in list(zip(factories.items(), last_factories.items()))]) * 10,
            'ore_to_factory': sum([max((f_info.cargo.ore - lf_info.cargo.ore), 0) for ((f_id, f_info), (lf_id, lf_info)) in list(zip(factories.items(), last_factories.items()))]) * 10,
            'ice_generation': (sum(list(stats['generation']['ice'].values())) - sum(list(last_stats['generation']['ice'].values()))) * 10,
            'ore_generation': (sum(list(stats['generation']['ore'].values())) - sum(list(last_stats['generation']['ore'].values()))) * 10,
            'power_generation': (stats['generation']['power']['FACTORY'] - last_stats['generation']['power']['FACTORY']) * 0,
            'water_generation': (stats['generation']['water'] - last_stats['generation']['water']) * 100,
            'metal_generation': (stats['generation']['metal'] - last_stats['generation']['metal']) * 100,
            # 'ice_transfer': stats['transfer']['ice'] - last_stats['transfer']['ice'],
            # 'ore_transfer': stats['transfer']['ore'] - last_stats['transfer']['ore'],
            'destroyed_rubble': (sum(list(stats['destroyed']['rubble'].values())) - sum(list(last_stats['destroyed']['rubble'].values()))) / 1,
        }
        if done:
            step_reward['ori_reward'] = ori_reward if ori_reward < 0 else ori_reward * 100
        for k in step_reward.keys():
            self.reward_collect[k] += step_reward[k]
        return sum(list(step_reward.values())), step_reward
