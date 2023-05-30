from typing import Any, Dict

import gym
import numpy as np
import numpy.typing as npt
from gym import spaces
# import matplotlib.pyplot as plt
from lux.config import EnvConfig
from wrappers.obs_space_conv import ObsSpace


class RwdTransfer:

    def __init__(self, env, env_cfg: EnvConfig, debug=False, density=True) -> None:
        self.env_cfg = env_cfg
        self.env = env
        self.obs_space = ObsSpace(env_cfg)
        self.debug = debug
        self.density = density
        self.reward_collect = {
            'ice_generation': 0,
            'ore_generation': 0,
            'water_generation': 0,
            'metal_generation': 0,
            # 'ice_to_factory': 0,
            # 'ore_to_factory': 0,
            # 'ice_transfer': 0,
            # 'ore_transfer': 0,
            'light_gene_collide': 0,
            'heavy_gene_collide': 0,
            'step': 0,
            'ori_reward': 0,
            'destroyed_rubble': 0,
            'power_generation': 0,
            # 'left_step_punish': 0
        }
        return

    def raw_to_wrap(self, ori_reward, done, last_stats, stats, last_factories, factories, last_units, units, last_obs, obs):
        step_reward = {
            'step': 1.0,
            # 'ice_to_factory': sum([max(-(u_info.cargo.ice - lu_info.cargo.ice), 0) for ((u_id, u_info), (lu_id, lu_info)) in list(zip(units.items(), last_units.items()))]) * 10,
            # 'ore_to_factory': sum([max(-(u_info.cargo.ore - lu_info.cargo.ore), 0) for ((u_id, u_info), (lu_id, lu_info)) in list(zip(units.items(), last_units.items()))]) * 10,
            'ice_generation': (sum(list(stats['generation']['ice'].values())) - sum(list(last_stats['generation']['ice'].values()))) * 10,
            'ore_generation': (sum(list(stats['generation']['ore'].values())) - sum(list(last_stats['generation']['ore'].values()))) * 10,
            'power_generation': (stats['generation']['power']['FACTORY'] - last_stats['generation']['power']['FACTORY']) * 0,
            'water_generation': (stats['generation']['water'] - last_stats['generation']['water']) * 100,
            'metal_generation': (stats['generation']['metal'] - last_stats['generation']['metal']) * 100,
            # 'ice_transfer': (stats['transfer']['ice'] - last_stats['transfer']['ice']) * 0.1,
            # 'ore_transfer': (stats['transfer']['ore'] - last_stats['transfer']['ore']) * 0.1,
            'destroyed_rubble': (sum(list(stats['destroyed']['rubble'].values())) - sum(list(last_stats['destroyed']['rubble'].values()))) * 0.01,
            'light_gene_collide': (sum([1 for u_id, u_info in units.items() if str(u_info.unit_type) == 'UnitType.LIGHT']) -
                                   sum([1 for u_id, u_info in last_units.items() if str(u_info.unit_type) == 'UnitType.LIGHT'])) * 10,
            'heavy_gene_collide': (sum([1 for u_id, u_info in units.items() if str(u_info.unit_type) == 'UnitType.HEAVY']) -
                                   sum([1 for u_id, u_info in last_units.items() if str(u_info.unit_type) == 'UnitType.HEAVY'])) * 100
        }
        # if self.reward_collect['ice_generation'] != 0:
        #     print('debug_water_transfer: ',
        #           np.sum([v.cargo.ice for k, v in units.items()]),
        #           np.sum(obs[self.obs_space.u_ice_dim_start]),
        #           np.sum([v.cargo.ice for k, v in factories.items()]),
        #           np.sum(obs[self.obs_space.f_ice_dim_start]),
        #           np.sum(obs[self.obs_space.f_water_dim_start]))
        if done:
            step_reward['ori_reward'] = ori_reward if ori_reward < 0 else ori_reward * 100
        for k in step_reward.keys():
            self.reward_collect[k] += step_reward[k]
        return sum(list(step_reward.values())), step_reward