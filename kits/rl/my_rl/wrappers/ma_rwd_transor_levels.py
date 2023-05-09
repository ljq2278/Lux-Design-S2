from typing import Any, Dict

import gym
import numpy as np
import numpy.typing as npt
from gym import spaces
# import matplotlib.pyplot as plt
from lux.config import EnvConfig
from wrappers.obs_space_levels import ObsSpaceUnit, ObsSpaceFactory
from wrappers.act_space_levels import ActSpaceUnit, ActSpaceFactory


class MaRwdTransorUnit():

    def __init__(self, env, env_cfg: EnvConfig, debug=False, density=True, save_water=50) -> None:
        self.env_cfg = env_cfg
        self.env = env
        self.debug = debug
        self.density = density
        self.save_water = save_water
        self.reward_collect = {
            'on the way target': 0,
            'leave the way target': 0,
            'dig out rubble on target': 0,
            'dig target': 0,
            'on the way rubble': 0,
            'leave the way rubble': 0,
            'on the way home with target': 0,
            'leave the way home with target': 0,
            'collision happen': 0,
            'transfer target': 0,
            'transfer ice': 0,
            'transfer ore': 0,
            'dig out target rubble': 0,
            'low power charged': 0,
        }
        return

    # def _if_in_factory(self, unit_info):
    #     if abs(unit_info[ObsSpaceUnit.target_factory_pos_start]) <= 1 \
    #             and abs(unit_info[ObsSpaceUnit.target_factory_pos_start + 1]) <= 1:
    #         return True
    #     return False

    def sg_to_ma(self, ori_reward, act, obs, next_obs, done, typ='HEAVY'):
        rewards = {}
        metrics = {}
        unit_ids = list(set(obs.keys()).union(set(next_obs.keys())))
        for unit_id in unit_ids:
            rewards[unit_id] = 0
            if done:
                rewards[unit_id] -= 100
                continue
            metrics[unit_id] = {}
            if unit_id not in next_obs.keys():  # it collide
                rewards[unit_id] -= 100
            elif unit_id not in obs.keys():  # it is new born
                pass
            else:
                metrics[unit_id]['transfered'] = next_obs[unit_id][ObsSpaceUnit.transfered_start]

                metrics[unit_id]['if_in_factory'] = bool(obs[unit_id][ObsSpaceUnit.is_at_home])
                metrics[unit_id]['if_next_in_factory'] = bool(next_obs[unit_id][ObsSpaceUnit.is_at_home])

                metrics[unit_id]['on_target'] = bool(obs[unit_id][ObsSpaceUnit.is_in_target])
                metrics[unit_id]['next_on_target'] = bool(next_obs[unit_id][ObsSpaceUnit.is_in_target])

                metrics[unit_id]['task_type'] = ObsSpaceUnit.int_to_task_type(obs[unit_id][ObsSpaceUnit.task_type_start])

                metrics[unit_id]['ice'] = obs[unit_id][ObsSpaceUnit.ice_dim_start]
                metrics[unit_id]['next_ice'] = next_obs[unit_id][ObsSpaceUnit.ice_dim_start]
                metrics[unit_id]['ice_changed'] = metrics[unit_id]['next_ice'] - metrics[unit_id]['ice']

                metrics[unit_id]['ore'] = obs[unit_id][ObsSpaceUnit.ore_dim_start]
                metrics[unit_id]['next_ore'] = next_obs[unit_id][ObsSpaceUnit.ore_dim_start]
                metrics[unit_id]['ore_changed'] = metrics[unit_id]['next_ore'] - metrics[unit_id]['ore']

                metrics[unit_id]['cargo'] = metrics[unit_id]['ice'] + metrics[unit_id]['ore']
                metrics[unit_id]['next_cargo'] = metrics[unit_id]['next_ice'] + metrics[unit_id]['next_ore']
                metrics[unit_id]['cargo_changed'] = metrics[unit_id]['next_cargo'] - metrics[unit_id]['cargo']

                metrics[unit_id]['power'] = obs[unit_id][ObsSpaceUnit.power_dim_start]
                metrics[unit_id]['next_power'] = next_obs[unit_id][ObsSpaceUnit.power_dim_start]
                metrics[unit_id]['power_changed'] = metrics[unit_id]['next_power'] - metrics[unit_id]['power']

                metrics[unit_id]['dis_to_factory'] = obs[unit_id][ObsSpaceUnit.home_dist]
                metrics[unit_id]['next_dis_to_factory'] = next_obs[unit_id][ObsSpaceUnit.home_dist]
                metrics[unit_id]['dis_to_factory_changed'] = metrics[unit_id]['next_dis_to_factory'] - metrics[unit_id]['dis_to_factory']

                metrics[unit_id]['dis_to_target'] = obs[unit_id][ObsSpaceUnit.target_dist]
                metrics[unit_id]['next_dis_to_target'] = next_obs[unit_id][ObsSpaceUnit.target_dist]
                metrics[unit_id]['dis_to_target_changed'] = metrics[unit_id]['next_dis_to_target'] - metrics[unit_id]['dis_to_target']

                metrics[unit_id]['curr_tile_rubble'] = obs[unit_id][int((ObsSpaceUnit.near_space_start + ObsSpaceUnit.near_space_start + ObsSpaceUnit.near_space) // 2)]
                metrics[unit_id]['next_curr_tile_rubble'] = next_obs[unit_id][int((ObsSpaceUnit.near_space_start + ObsSpaceUnit.near_space_start + ObsSpaceUnit.near_space) // 2)]
                metrics[unit_id]['rubble_changed'] = metrics[unit_id]['next_curr_tile_rubble'] - metrics[unit_id]['curr_tile_rubble']
                if self.debug:
                    print('############################################### unit debug start #########################################################################################################')
                    # print(metrics[unit_id]['rubble_changed'], self.env_cfg.MAX_RUBBLE)
                rewards[unit_id] += metrics[unit_id]['transfered'] / 10

                if 0 <= act[unit_id] < ActSpaceUnit.move_dim_high:
                    if ObsSpaceUnit.move_target_rubble(obs[unit_id], act[unit_id]) > self.env_cfg.MAX_RUBBLE:  ############################################### collision happen
                        rwd = -60
                        rewards[unit_id] += rwd
                        next_obs[unit_id][ObsSpaceUnit.transfered_start] += rwd
                        self.reward_collect['collision happen'] += 1
                        if self.debug:
                            print('collision happen')

                if metrics[unit_id]['task_type'] == 'rubble':
                    if self.density and metrics[unit_id]['dis_to_target_changed'] < 0:  ###################################################################### on the way rubble
                        rwd = 1
                        rewards[unit_id] += rwd
                        self.reward_collect['on the way rubble'] += 1
                        if self.debug:
                            print('on the way rubble')

                    if self.density and metrics[unit_id]['dis_to_target_changed'] > 0:  ###################################################################### leave the way rubble
                        rwd = -1.1
                        rewards[unit_id] += rwd
                        self.reward_collect['leave the way rubble'] += 1
                        if self.debug:
                            print('leave the way rubble')

                    if metrics[unit_id]['on_target'] and metrics[unit_id]['next_on_target']:  ################################################################### dig rubble reward
                        if metrics[unit_id]['rubble_changed'] < 0:
                            rwd = -metrics[unit_id]['rubble_changed']
                            next_obs[unit_id][ObsSpaceUnit.transfered_start] += -metrics[unit_id]['rubble_changed']
                            metrics[unit_id]['transfered'] += -metrics[unit_id]['rubble_changed']
                            rewards[unit_id] += rwd
                            self.reward_collect['dig out target rubble'] += 1
                            if self.debug:
                                print('dig out target rubble')

                elif metrics[unit_id]['task_type'] == 'ice' or metrics[unit_id]['task_type'] == 'ore':

                    if self.density and metrics[unit_id]['dis_to_target_changed'] < 0 and metrics[unit_id][metrics[unit_id]['task_type']] == 0:  ############## on the way target
                        rwd = 1
                        rewards[unit_id] += rwd
                        self.reward_collect['on the way target'] += 1
                        if self.debug:
                            print('on the way target')

                    if self.density and metrics[unit_id]['dis_to_target_changed'] > 0 and metrics[unit_id][metrics[unit_id]['task_type']] == 0:  ############## leave the way target
                        rwd = -1.1
                        rewards[unit_id] += rwd
                        self.reward_collect['leave the way target'] += 1
                        if self.debug:
                            print('leave the way target')

                    if metrics[unit_id]['on_target'] and metrics[unit_id]['next_on_target'] and metrics[unit_id][metrics[unit_id]['task_type']] < 200:  ########### dig reward
                        if metrics[unit_id]['rubble_changed'] < 0:
                            rwd = 1.5
                            rewards[unit_id] += rwd
                            self.reward_collect['dig out rubble on target'] += 1
                            if self.debug:
                                print('dig out rubble on target')
                        elif metrics[unit_id][metrics[unit_id]['task_type'] + '_changed'] > 0:
                            factor = 0.1
                            rwd = metrics[unit_id][metrics[unit_id]['task_type'] + '_changed'] * factor
                            rewards[unit_id] += rwd
                            self.reward_collect['dig target'] += 1
                            if self.debug:
                                print('dig target')

                    if self.density and metrics[unit_id]['dis_to_factory_changed'] < 0 and \
                            metrics[unit_id][metrics[unit_id]['task_type']] > 0 and not metrics[unit_id]['if_in_factory']:  ########### on the way home with target
                        factor = 0.1
                        rwd = metrics[unit_id][metrics[unit_id]['task_type']] * factor
                        rewards[unit_id] += rwd
                        self.reward_collect['on the way home with target'] += 1
                        if self.debug:
                            print('on the way home with target')

                    if self.density and metrics[unit_id]['dis_to_factory_changed'] > 0 and \
                            metrics[unit_id][metrics[unit_id]['task_type']] > 0 and metrics[unit_id]['if_in_factory']:  ########### leave the way home with target
                        factor = -0.11
                        rwd = metrics[unit_id][metrics[unit_id]['task_type']] * factor
                        rewards[unit_id] += rwd
                        self.reward_collect['leave the way home with target'] += 1
                        if self.debug:
                            print('leave the way home with target')

                    if metrics[unit_id]['if_in_factory']:  ######################################################################################################### transfer target
                        if metrics[unit_id][metrics[unit_id]['task_type'] + '_changed'] < 0:
                            factor = 1
                            rwd = -metrics[unit_id][metrics[unit_id]['task_type'] + '_changed'] * factor
                            rewards[unit_id] += rwd
                            self.reward_collect['transfer target'] += 1
                            metrics[unit_id]['transfered'] += -metrics[unit_id][metrics[unit_id]['task_type'] + '_changed']
                            next_obs[unit_id][ObsSpaceUnit.transfered_start] += -metrics[unit_id][metrics[unit_id]['task_type'] + '_changed']
                            if self.debug:
                                print('transfered')
                            if metrics[unit_id]['task_type'] == 'ice':
                                self.reward_collect['transfer ice'] += 1
                            elif metrics[unit_id]['task_type'] == 'ore':
                                self.reward_collect['transfer ore'] += 1
                else:
                    pass
                if self.density and metrics[unit_id]['power_changed'] > 0 and metrics[unit_id]['power'] < self.env_cfg.ROBOTS[typ].BATTERY_CAPACITY * 0.2:  ############ low power charge reward
                    rwd = 0.002 * metrics[unit_id]['power_changed']
                    rewards[unit_id] += rwd
                    self.reward_collect['low power charged'] += 1

                if self.debug:
                    print(unit_id)
                    print(obs[unit_id])
                    print(next_obs[unit_id])
                    print(act[unit_id])
                    print(np.array(obs[unit_id][ObsSpaceUnit.near_space_start:ObsSpaceUnit.near_space_start + ObsSpaceUnit.near_space])
                          .reshape([int(np.sqrt(ObsSpaceUnit.near_space)), int(np.sqrt(ObsSpaceUnit.near_space))]))
                    print(np.array(next_obs[unit_id][ObsSpaceUnit.near_space_start:ObsSpaceUnit.near_space_start + ObsSpaceUnit.near_space])
                          .reshape([int(np.sqrt(ObsSpaceUnit.near_space)), int(np.sqrt(ObsSpaceUnit.near_space))]))
                    print(rewards[unit_id])
                    # print(self.reward_collect)
                    print('################################################ unit debug end #########################################################################################################')
        return rewards, next_obs


class MaRwdTransorFactory():
    def __init__(self, env, env_cfg: EnvConfig, debug=False, density=True, save_water=50) -> None:
        self.env_cfg = env_cfg
        self.env = env
        self.debug = debug
        self.density = density
        self.save_water = save_water
        self.reward_collect = {
            'water add': 0,
            'metal add': 0,
            'ice add': 0,
            'ore add': 0,
            'heavy robot made': 0,
            'light robot made': 0,
            'water lichen': 0,
        }
        return

    def sg_to_ma(self, ori_reward, act, obs, next_obs, done, factory_info, ice_map=None, ore_map=None, raw_obs=None, typ='HEAVY', target='ice'):
        rewards = {}
        metrics = {}
        factory_ids = list(set(obs.keys()).union(set(next_obs.keys())))
        for f_id in factory_ids:
            rewards[f_id] = 0
            if done:
                rewards[f_id] -= 100
                continue
            metrics[f_id] = {}
            if f_id not in next_obs.keys():  # it collide
                rewards[f_id] -= 10
            elif f_id not in obs.keys():  # it is new born
                pass
            else:

                metrics[f_id]['ice'] = obs[f_id][ObsSpaceFactory.ice_dim_start]
                metrics[f_id]['next_ice'] = next_obs[f_id][ObsSpaceFactory.ice_dim_start]
                metrics[f_id]['ice_changed'] = metrics[f_id]['next_ice'] - metrics[f_id]['ice']

                metrics[f_id]['ore'] = obs[f_id][ObsSpaceFactory.ore_dim_start]
                metrics[f_id]['next_ore'] = next_obs[f_id][ObsSpaceFactory.ore_dim_start]
                metrics[f_id]['ore_changed'] = metrics[f_id]['next_ore'] - metrics[f_id]['ore']
                if self.debug:
                    print('############################################## factory debug start ########################################################################################################')
                if act[f_id] == ActSpaceFactory.water_lichen_high - 1: ####################### water lichen
                    rwd = len(factory_info[f_id].connected_lichen_positions)
                    rewards[f_id] += rwd
                    self.reward_collect['water lichen'] += len(factory_info[f_id].connected_lichen_positions)
                    if self.debug:
                        print('water lichen')

                if metrics[f_id]['ice_changed'] > 0:  ########################################## ice add
                    rwd = metrics[f_id]['ice_changed']
                    rewards[f_id] += rwd
                    self.reward_collect['ice add'] += 1
                    if self.debug:
                        print('ice add')

                if metrics[f_id]['ore_changed'] > 0:  ########################################## ore add
                    rwd = metrics[f_id]['ore_changed']
                    rewards[f_id] += rwd
                    self.reward_collect['ore add'] += 1
                    if self.debug:
                        print('ore add')

                if act[f_id] == ActSpaceFactory.build_heavy_high - 1:  ######################## add heavy
                    rwd = 100
                    rewards[f_id] += rwd
                    self.reward_collect['heavy robot made'] += 1
                    if self.debug:
                        print('heavy robot made')
                if self.debug:
                    print('############################################ factory debug end #########################################################################################################')
        return rewards
