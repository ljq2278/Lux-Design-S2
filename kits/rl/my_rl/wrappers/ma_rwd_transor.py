from typing import Any, Dict

import gym
import numpy as np
import numpy.typing as npt
from gym import spaces
# import matplotlib.pyplot as plt
from lux.config import EnvConfig
from wrappers.obs_space import ObsSpace


# act_dict = {
#     'move_a': 0,  # 0~3
#     'move_b': 1,  # 0~3
#     'move_c': 2,  # 0~3
#     'move_d': 3,  # 0~3
#     'transfer_ice': 4,  # 4
#     'transfer_ore': 5,  # 5
#     'pick_up': 6,  # 6
#     'dig': 7,  # 7
#     'no_op': 8,  # 8
# }


class MaRwdTransor():

    def __init__(self, env, env_cfg: EnvConfig, debug=False, density=True, save_water=50) -> None:
        self.env_cfg = env_cfg
        self.env = env
        self.debug = debug
        self.density = density
        self.save_water = save_water
        self.reward_collect = {
            'leave the way home with ice': 0,
            'on the way home with ice': 0,
            'leave the way home with ore': 0,
            'on the way home with ore': 0,
            'leave the way ice with low cargo': 0,
            'on the way ice with low cargo': 0,
            'leave the way ore with low cargo': 0,
            'on the way ore with low cargo': 0,
            'get to ice tile': 0,
            'get to ore tile': 0,
            'prepare for dig': 0,
            'dig out rubble on ice': 0,
            'dig ice success': 0,
            'dig out rubble on ore': 0,
            'dig ore success': 0,
            'want to leave factory with ice': 0,
            'want to leave factory with ore': 0,
            'return factory with ices': 0,
            'return factory with ores': 0,
            'transfer ice success': 0,
            'transfer ore success': 0,
            'low power charged': 0,
        }
        return

    def _if_in_factory(self, unit_info):
        if abs(unit_info[ObsSpace.nearest_factory_pos_start]) <= 1 \
                and abs(unit_info[ObsSpace.nearest_factory_pos_start + 1]) <= 1:
            return True
        return False

    def _if_on_ore(self, pos, ore_board):
        if ore_board[pos[0], pos[1]] == 1:
            return True
        return False

    def _if_on_ice(self, pos, ice_board):
        if ice_board[pos[0], pos[1]] == 1:
            return True
        return False

    def _ice_reward_factor(self, f_ice, f_water):
        return 1 if f_water < self.save_water else 0
        # return np.log(31 / (f_water + 1))
        # return 1 / (f_water / 25 + 1) + 0.4
        # return 0

    def _ore_reward_factor(self, f_ore, f_metal, f_water):
        return 0 if f_water < self.save_water else 1
        # return np.log((f_water + 1) / 31)
        # return np.log(300 / (f_ore + 100)) + np.log(100 / (f_metal + 10))
        # return 1 / (f_ore / 1000 + 1) + 1 / (f_metal / 50 + 1)

    def sg_to_ma(self, ori_reward, act, obs, next_obs, done, ice_map=None, ore_map=None, raw_obs=None, typ='HEAVY'):
        rewards = {}
        metrics = {}
        unit_ids = list(set(obs.keys()).union(set(next_obs.keys())))
        # global_reward = (len(unit_ids) - 2) * 1
        for unit_id in unit_ids:
            rewards[unit_id] = 0
            if done:
                rewards[unit_id] -= 100
                continue
            metrics[unit_id] = {}
            if unit_id not in next_obs.keys():  # it collide
                rewards[unit_id] -= 10
            elif unit_id not in obs.keys():  # it is new born
                pass
            else:
                if obs[unit_id][ObsSpace.nearest_factory_pos_start] == -100:  ####################### no factory
                    rewards[unit_id] -= 100
                    continue

                metrics[unit_id]['min_factory_water'] = obs[unit_id][ObsSpace.nearest_factory_water_start]
                metrics[unit_id]['if_in_factory'] = self._if_in_factory(obs[unit_id])
                metrics[unit_id]['if_next_in_factory'] = self._if_in_factory(next_obs[unit_id])

                metrics[unit_id]['on_ice'] = bool(abs(obs[unit_id][ObsSpace.nearest_ice_pos_start]) +
                                                  abs(obs[unit_id][ObsSpace.nearest_ice_pos_start + 1]) == 0)
                metrics[unit_id]['next_on_ice'] = bool(abs(next_obs[unit_id][ObsSpace.nearest_ice_pos_start]) +
                                                       abs(next_obs[unit_id][ObsSpace.nearest_ice_pos_start + 1]) == 0)
                metrics[unit_id]['on_ore'] = bool(abs(obs[unit_id][ObsSpace.nearest_ore_pos_start]) +
                                                  abs(obs[unit_id][ObsSpace.nearest_ore_pos_start + 1]) == 0)
                metrics[unit_id]['next_on_ore'] = bool(abs(next_obs[unit_id][ObsSpace.nearest_ore_pos_start]) +
                                                       abs(next_obs[unit_id][ObsSpace.nearest_ore_pos_start + 1]) == 0)

                metrics[unit_id]['ice'], metrics[unit_id]['ice_norm'] = obs[unit_id][3], obs[unit_id][3] / self.env_cfg.ROBOTS[typ].CARGO_SPACE
                metrics[unit_id]['next_ice'], metrics[unit_id]['next_ice_norm'] = next_obs[unit_id][3], next_obs[unit_id][3] / self.env_cfg.ROBOTS[typ].CARGO_SPACE
                metrics[unit_id]['ice_changed'] = metrics[unit_id]['next_ice'] - metrics[unit_id]['ice']
                metrics[unit_id]['ice_changed_norm'] = metrics[unit_id]['ice_changed'] / self.env_cfg.ROBOTS[typ].CARGO_SPACE
                metrics[unit_id]['ore'], metrics[unit_id]['ore_norm'] = obs[unit_id][4], obs[unit_id][4] / self.env_cfg.ROBOTS[typ].CARGO_SPACE
                metrics[unit_id]['next_ore'], metrics[unit_id]['next_ore_norm'] = next_obs[unit_id][4], next_obs[unit_id][4] / self.env_cfg.ROBOTS[typ].CARGO_SPACE
                metrics[unit_id]['ore_changed'] = metrics[unit_id]['next_ore'] - metrics[unit_id]['ore']
                metrics[unit_id]['ore_changed_norm'] = metrics[unit_id]['ore_changed'] / self.env_cfg.ROBOTS[typ].CARGO_SPACE

                metrics[unit_id]['cargo'] = metrics[unit_id]['ice'] + metrics[unit_id]['ore']
                metrics[unit_id]['next_cargo'] = metrics[unit_id]['next_ice'] + metrics[unit_id]['next_ore']
                metrics[unit_id]['cargo_changed'] = metrics[unit_id]['next_cargo'] - metrics[unit_id]['cargo']

                metrics[unit_id]['power'] = obs[unit_id][2]
                metrics[unit_id]['next_power'] = next_obs[unit_id][2]
                metrics[unit_id]['power_changed'] = metrics[unit_id]['next_power'] - metrics[unit_id]['power']

                metrics[unit_id]['min_dis_to_factory'] = \
                    obs[unit_id][ObsSpace.nearest_factory_pos_start] ** 2 \
                    + obs[unit_id][ObsSpace.nearest_factory_pos_start + 1] ** 2

                metrics[unit_id]['next_min_dis_to_factory'] = \
                    next_obs[unit_id][ObsSpace.nearest_factory_pos_start] ** 2 \
                    + next_obs[unit_id][ObsSpace.nearest_factory_pos_start + 1] ** 2

                metrics[unit_id]['min_dis_to_ice'] = \
                    obs[unit_id][ObsSpace.nearest_ice_pos_start] ** 2 \
                    + obs[unit_id][ObsSpace.nearest_ice_pos_start + 1] ** 2

                metrics[unit_id]['next_min_dis_to_ice'] = \
                    next_obs[unit_id][ObsSpace.nearest_ice_pos_start] ** 2 \
                    + next_obs[unit_id][ObsSpace.nearest_ice_pos_start + 1] ** 2

                metrics[unit_id]['min_dis_to_ore'] = \
                    obs[unit_id][ObsSpace.nearest_ore_pos_start] ** 2 \
                    + obs[unit_id][ObsSpace.nearest_ore_pos_start + 1] ** 2

                metrics[unit_id]['next_min_dis_to_ore'] = \
                    next_obs[unit_id][ObsSpace.nearest_ore_pos_start] ** 2 \
                    + next_obs[unit_id][ObsSpace.nearest_ore_pos_start + 1] ** 2

                metrics[unit_id]['curr_tile_rubble'] = obs[unit_id][
                    int((ObsSpace.near_space_start + ObsSpace.near_space_start + ObsSpace.near_space) // 2)]
                metrics[unit_id]['next_curr_tile_rubble'] = next_obs[unit_id][
                    int((ObsSpace.near_space_start + ObsSpace.near_space_start + ObsSpace.near_space) // 2)]
                metrics[unit_id]['rubble_changed'] = \
                    metrics[unit_id]['next_curr_tile_rubble'] - metrics[unit_id]['curr_tile_rubble']

                metrics[unit_id]['near_factory_ice'] = obs[unit_id][ObsSpace.nearest_factory_ice_start]
                metrics[unit_id]['near_factory_water'] = obs[unit_id][ObsSpace.nearest_factory_water_start]
                metrics[unit_id]['ice_factor'] = self._ice_reward_factor(metrics[unit_id]['near_factory_ice'], metrics[unit_id]['near_factory_water'])
                metrics[unit_id]['near_factory_ore'] = obs[unit_id][ObsSpace.nearest_factory_ore_start]
                metrics[unit_id]['near_factory_metal'] = obs[unit_id][ObsSpace.nearest_factory_metal_start]
                metrics[unit_id]['ore_factor'] = self._ore_reward_factor(metrics[unit_id]['near_factory_ore'], metrics[unit_id]['near_factory_metal'], metrics[unit_id]['near_factory_water'])

                rewards[unit_id] += (metrics[unit_id]['near_factory_ice'] + metrics[unit_id]['near_factory_ore'])

                if self.debug:
                    print(obs[unit_id])
                    print(next_obs[unit_id])
                if self.density and metrics[unit_id]['ice'] > 0:  ############################################################# on the way home with ice
                    base = 20
                    factor = 20
                    dist_diff = metrics[unit_id]['next_min_dis_to_factory'] - metrics[unit_id]['min_dis_to_factory']
                    factor2 = 1 if dist_diff < 0 else (0 if dist_diff == 0 else 0)
                    rwd = metrics[unit_id]['ice_factor'] * (base + factor * metrics[unit_id]['ice_norm']) * factor2
                    rewards[unit_id] += rwd
                    if rwd < 0:
                        self.reward_collect['leave the way home with ice'] += 1
                    elif rwd > 0:
                        self.reward_collect['on the way home with ice'] += 1

                if self.density and metrics[unit_id]['ore'] > 0:  ############################################################# on the way home with ore
                    base = 20
                    factor = 20
                    dist_diff = metrics[unit_id]['next_min_dis_to_factory'] - metrics[unit_id]['min_dis_to_factory']
                    factor2 = 1 if dist_diff < 0 else (0 if dist_diff == 0 else 0)
                    rwd = metrics[unit_id]['ore_factor'] * (base + factor * metrics[unit_id]['ore_norm']) * factor2
                    rewards[unit_id] += rwd
                    if rwd < 0:
                        self.reward_collect['leave the way home with ore'] += 1
                    elif rwd > 0:
                        self.reward_collect['on the way home with ore'] += 1

                if self.density and metrics[unit_id]['ice'] == 0:  ############################################################ on the way ice with low cargo
                    dist_diff = metrics[unit_id]['next_min_dis_to_ice'] - metrics[unit_id]['min_dis_to_ice']
                    rwd = metrics[unit_id]['ice_factor'] * (0.1 if dist_diff < 0 else (0 if dist_diff == 0 else 0))
                    rewards[unit_id] += rwd
                    if rwd < 0:
                        self.reward_collect['leave the way ice with low cargo'] += 1
                    elif rwd > 0:
                        self.reward_collect['on the way ice with low cargo'] += 1

                if self.density and metrics[unit_id]['ore'] == 0:  ############################################################ on the way ore with low cargo
                    dist_diff = metrics[unit_id]['next_min_dis_to_ore'] - metrics[unit_id]['min_dis_to_ore']
                    rwd = metrics[unit_id]['ore_factor'] * (0.08 if dist_diff < 0 else (0 if dist_diff == 0 else 0))
                    rewards[unit_id] += rwd
                    if rwd < 0:
                        self.reward_collect['leave the way ore with low cargo'] += 1
                    elif rwd > 0:
                        self.reward_collect['on the way ore with low cargo'] += 1

                if self.density and act[unit_id] < 4 and \
                        (not metrics[unit_id]['on_ice']) and metrics[unit_id]['next_on_ice'] and metrics[unit_id]['cargo'] < 50:  ######### get to ice tile low cargo
                    rwd = metrics[unit_id]['ice_factor'] * 5
                    rewards[unit_id] += rwd
                    self.reward_collect['get to ice tile'] += 1

                if self.density and act[unit_id] < 4 and \
                        (not metrics[unit_id]['on_ore']) and metrics[unit_id]['next_on_ore'] and metrics[unit_id]['cargo'] < 50:  ########## get to ore tile low cargo
                    rwd = metrics[unit_id]['ore_factor'] * 3
                    rewards[unit_id] += rwd
                    self.reward_collect['get to ore tile'] += 1

                if self.density and (act[unit_id] == 8 and (metrics[unit_id]['on_ice'] or metrics[unit_id]['on_ore'])
                                     and metrics[unit_id]['power'] < 1.2 * self.env_cfg.ROBOTS[typ].DIG_COST):  ################################### prepare for dig
                    rwd = 1
                    rewards[unit_id] += rwd
                    self.reward_collect['prepare for dig'] += 1
                    # if self.debug:
                    #     print(unit_id, ' prepare for dig ', 2)

                if act[unit_id] == 7 and metrics[unit_id]['on_ice'] and metrics[unit_id]['rubble_changed'] < 0:  ########################## dig out rubble on ice
                    rwd = metrics[unit_id]['ice_factor'] * 8
                    rewards[unit_id] += rwd
                    self.reward_collect['dig out rubble on ice'] += 1
                    if self.debug:
                        print(unit_id, ' dig out rubble on ice ', rwd)

                if act[unit_id] == 7 and metrics[unit_id]['on_ore'] and metrics[unit_id]['rubble_changed'] < 0:  ########################## dig out rubble on ore
                    rwd = metrics[unit_id]['ore_factor'] * 6
                    rewards[unit_id] += rwd
                    self.reward_collect['dig out rubble on ore'] += 1
                    if self.debug:
                        print(unit_id, ' dig out rubble on ore ', rwd)

                if act[unit_id] == 7 and metrics[unit_id]['ice_changed'] > 0 and metrics[unit_id]['ice'] < 100 \
                        and metrics[unit_id]['near_factory_ice'] < 100:  ####################################################### dig ice success
                    # factor = (metrics[unit_id]['ice_norm'] - 1) ** 100
                    factor = 1
                    rwd = metrics[unit_id]['ice_factor'] * factor * (metrics[unit_id]['ice_changed'])
                    rewards[unit_id] += rwd
                    self.reward_collect['dig ice success'] += 1
                    if self.debug:
                        print(unit_id, ' dig ice success ', rwd, metrics[unit_id]['power'])

                if act[unit_id] == 7 and metrics[unit_id]['ore_changed'] > 0 and metrics[unit_id]['ore'] < 100 \
                        and metrics[unit_id]['near_factory_ore'] < 100:  ####################################################### dig ore success
                    # factor = (metrics[unit_id]['ore_norm'] - 1) ** 100
                    factor = 0.8
                    rwd = metrics[unit_id]['ore_factor'] * factor * (metrics[unit_id]['ore_changed'])
                    rewards[unit_id] += rwd
                    self.reward_collect['dig ore success'] += 1
                    if self.debug:
                        print(unit_id, ' dig ore success ', rwd, metrics[unit_id]['power'])

                if self.density and act[unit_id] < 4 and metrics[unit_id]['if_in_factory'] and metrics[unit_id]['ice'] > 0:  ############ want to leave factory with ice
                    base = -20
                    factor = -20
                    rwd = metrics[unit_id]['ice_factor'] * (base + factor * metrics[unit_id]['ice_norm'])
                    rewards[unit_id] += rwd
                    self.reward_collect['want to leave factory with ice'] += 1
                    if self.debug:
                        print(unit_id, ' want to leave factory with ice ', rwd)

                if self.density and act[unit_id] < 4 and metrics[unit_id]['if_in_factory'] and metrics[unit_id]['ore'] > 0:  ############ want to leave factory with ore
                    base = -20
                    factor = -20
                    rwd = metrics[unit_id]['ore_factor'] * (base + factor * metrics[unit_id]['ore_norm'])
                    rewards[unit_id] += rwd
                    self.reward_collect['want to leave factory with ore'] += 1
                    if self.debug:
                        print(unit_id, ' want to leave factory with ore ', rwd)

                if self.density and act[unit_id] < 4 and not metrics[unit_id]['if_in_factory'] \
                        and metrics[unit_id]['if_next_in_factory'] and metrics[unit_id]['ice'] > 0:  ################################## return factory with ices
                    base = 30
                    factor = 30
                    rwd = metrics[unit_id]['ice_factor'] * (base + factor * metrics[unit_id]['ice_norm'])
                    rewards[unit_id] += rwd
                    self.reward_collect['return factory with ices'] += 1
                    if self.debug:
                        print(unit_id, ' return to factory with ice success ', rwd)

                if self.density and act[unit_id] < 4 and not metrics[unit_id]['if_in_factory'] \
                        and metrics[unit_id]['if_next_in_factory'] and metrics[unit_id]['ore'] > 0:  ################################### return factory with ores
                    base = 20
                    factor = 20
                    rwd = metrics[unit_id]['ore_factor'] * (base + factor * metrics[unit_id]['ore_norm'])
                    rewards[unit_id] += rwd
                    self.reward_collect['return factory with ores'] += 1
                    if self.debug:
                        print(unit_id, ' return to factory with ore success ', rwd)

                if act[unit_id] == 4 and metrics[unit_id]['if_in_factory'] and metrics[unit_id]['ice_changed'] < 0:  ################## transfer ice success
                    base = 50
                    factor = -50
                    rwd = metrics[unit_id]['ice_factor'] * (base + factor * metrics[unit_id]['ice_changed_norm'])
                    rewards[unit_id] += rwd
                    self.reward_collect['transfer ice success'] += 1
                    if self.debug:
                        print(unit_id, ' transfer ice success ', rwd)

                if act[unit_id] == 5 and metrics[unit_id]['if_in_factory'] and metrics[unit_id]['ore_changed'] < 0:  ################### transfer ore success
                    base = 40
                    factor = -40
                    rwd = metrics[unit_id]['ore_factor'] * (base + factor * metrics[unit_id]['ore_changed_norm'])
                    rewards[unit_id] += rwd
                    self.reward_collect['transfer ore success'] += 1
                    if self.debug:
                        print(unit_id, ' transfer ore success ', rwd)

                if self.density and metrics[unit_id]['power_changed'] > 0 and \
                        metrics[unit_id]['power'] < self.env_cfg.ROBOTS[
                    typ].BATTERY_CAPACITY * 0.2:  ############################################################################# low power charge reward
                    rwd = 0.002 * metrics[unit_id]['power_changed']
                    rewards[unit_id] += rwd
                    self.reward_collect['low power charged'] += 1
        return rewards
