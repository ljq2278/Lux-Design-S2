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
            # 'leave the way rubble': 0,
            # 'on the way home with target': 0,
            # 'leave the way home with target': 0,
            # 'collision happen': 0,
            # 'transfer target': 0,
            # 'transfer ice': 0,
            # 'transfer ore': 0,
            # 'dig out target rubble': 0,
            # 'low power charged': 0,
            'rewards': 0
        }
        return

    def raw_to_wrap(self, ori_reward, done, last_stats, stats):
        if done:
            self.reward_collect['rewards'] += ori_reward
            return ori_reward
        else:
            self.reward_collect['rewards'] += 10
            ice_generation = (stats['generation']['ice']['HEAVY'] + stats['generation']['ice']['LIGHT']) - \
                             (last_stats['generation']['ice']['HEAVY'] + last_stats['generation']['ice']['LIGHT'])
            ore_generation = (stats['generation']['ore']['HEAVY'] + stats['generation']['ore']['LIGHT']) - \
                             (last_stats['generation']['ore']['HEAVY'] + last_stats['generation']['ore']['LIGHT'])
            power_generation = stats['generation']['power']['FACTORY'] - last_stats['generation']['power']['FACTORY']
            water_generation = stats['generation']['water'] - last_stats['generation']['water']
            metal_generation = stats['generation']['metal'] - last_stats['generation']['metal']
            # ore_transfer = stats['transfer']['ore'] - last_stats['transfer']['ore']

            self.reward_collect['ice_generation'] += ice_generation
            self.reward_collect['ore_generation'] += ore_generation
            self.reward_collect['power_generation'] += power_generation
            self.reward_collect['water_generation'] += water_generation
            self.reward_collect['metal_generation'] += metal_generation

            return 10 + ice_generation + ore_generation + power_generation + water_generation + metal_generation
