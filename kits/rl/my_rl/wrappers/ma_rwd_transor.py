from typing import Any, Dict

import gym
import numpy as np
import numpy.typing as npt
from gym import spaces
# import matplotlib.pyplot as plt
from lux.config import EnvConfig

# def show(env):
#     img = env.render("rgb_array", width=640, height=640)
#     plt.imshow(img)
#     plt.show()

act_dict = {
    'move_l': 0,  # 0~3
    'move_u': 1,  # 0~3
    'move_r': 2,  # 0~3
    'move_d': 3,  # 0~3
    'transfer_ice': 4,  # 4
    'transfer_ore': 5,  # 5
    'pick_up': 6,  # 6
    'dig': 7,  # 7
    'no_op': 8,  # 8
}


class MaRwdTransor():

    def __init__(self, env, env_cfg: EnvConfig, debug=False) -> None:
        self.env_cfg = env_cfg
        self.env = env
        self.debug = debug
        return

    def _if_in_factory(self, unit_info):
        if (abs(unit_info[46]) <= 1 and abs(unit_info[47]) <= 1) or \
                (abs(unit_info[48]) <= 1 and abs(unit_info[49]) <= 1) or \
                (abs(unit_info[50]) <= 1 and abs(unit_info[51]) <= 1):
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

    def sg_to_ma(self, ori_reward, act, obs, next_obs, ice_map, ore_map):
        # show(self.env)
        rewards = {}
        metrics = {}
        unit_ids = list(set(obs.keys()).union(set(next_obs.keys())))
        for unit_id in unit_ids:
            rewards[unit_id] = 0
            if unit_id not in next_obs.keys():  # it collide
                rewards[unit_id] -= 10
            elif unit_id not in obs.keys():  # it is new born
                pass
            else:
                metrics[unit_id] = {}
                metrics[unit_id]['if_in_factory'] = self._if_in_factory(obs[unit_id])
                metrics[unit_id]['if_next_in_factory'] = self._if_in_factory(next_obs[unit_id])
                metrics[unit_id]['ice'] = obs[unit_id][3]
                metrics[unit_id]['ore'] = obs[unit_id][4]

                if act[unit_id] < 4 and ice_map[obs[unit_id][0], obs[unit_id][1]] != 1 and \
                        ice_map[next_obs[unit_id][0], next_obs[unit_id][1]] == 1:  ################## get to ice tile
                    rewards[unit_id] += 5
                    if self.debug:
                        print(unit_id, ' get to ice tile ',5)

                if act[unit_id] < 4 and ore_map[obs[unit_id][0], obs[unit_id][1]] != 1 and \
                        ore_map[next_obs[unit_id][0], next_obs[unit_id][1]] == 1:  ################### get to ore tile
                    rewards[unit_id] += 2
                    if self.debug:
                        print(unit_id, ' get to ore tile ',2)

                if act[unit_id] == 7 \
                        and not (next_obs[unit_id][3] - obs[unit_id][3] > 0) \
                        and not (next_obs[unit_id][4] - obs[unit_id][4] > 0):  ################## dig ice or ore failed
                    rewards[unit_id] -= 0.1
                    if self.debug:
                        print(unit_id, ' dig ice or ore failed ', -0.1)

                if act[unit_id] == 7 and next_obs[unit_id][3] - obs[unit_id][3] > 0:  ################# dig ice success
                    rewards[unit_id] += 1 * (next_obs[unit_id][3] - obs[unit_id][3])
                    if self.debug:
                        print(unit_id, ' dig ice success ', 1 * (next_obs[unit_id][3] - obs[unit_id][3]))

                if act[unit_id] == 7 and next_obs[unit_id][4] - obs[unit_id][4] > 0:  ############## dig ore success
                    rewards[unit_id] += 0.5 * (next_obs[unit_id][4] - obs[unit_id][4])
                    if self.debug:
                        print(unit_id, ' dig ore success ', 0.5 * (next_obs[unit_id][4] - obs[unit_id][4]))

                if not metrics[unit_id]['if_in_factory'] and act[unit_id] < 4 \
                        and metrics[unit_id]['if_next_in_factory'] \
                        and metrics[unit_id]['ice'] > 0:  ################################ return factory with ices
                    rewards[unit_id] += 0.2 * metrics[unit_id]['ice']
                    if self.debug:
                        print(unit_id, ' return to factory with ice success ', 0.2 * metrics[unit_id]['ice'])

                if not metrics[unit_id]['if_in_factory'] and act[unit_id] < 4 \
                        and metrics[unit_id]['if_next_in_factory'] \
                        and metrics[unit_id]['ore'] > 0:  #################################### return factory with ores
                    rewards[unit_id] += 0.1 * metrics[unit_id]['ore']
                    if self.debug:
                        print(unit_id, ' return to factory with ore success ', 0.1 * metrics[unit_id]['ore'])

                if act[unit_id] == 4 and metrics[unit_id]['if_in_factory'] \
                        and obs[unit_id][3] - next_obs[unit_id][3] > 0:  ######################### transfer ice success
                    rewards[unit_id] += 1 * (obs[unit_id][3] - next_obs[unit_id][3])
                    if self.debug:
                        print(unit_id, ' transfer ice success ', 0.1 * (obs[unit_id][3] - next_obs[unit_id][3]))

                if act[unit_id] == 4 and not metrics[unit_id]['if_in_factory']:  ############## transfer ice failed
                    rewards[unit_id] -= 0.1
                    if self.debug:
                        print(unit_id, ' transfer ice failed, power cost ', next_obs[unit_id][2] - obs[unit_id][2])
                              # ' ice cost ', obs[unit_id][3] - next_obs[unit_id][3])

                if act[unit_id] == 5 and metrics[unit_id]['if_in_factory'] \
                        and obs[unit_id][4] - next_obs[unit_id][4] > 0:  ###################### transfer ore success
                    rewards[unit_id] += 0.5 * (obs[unit_id][4] - next_obs[unit_id][4])
                    if self.debug:
                        print(unit_id, ' transfer ore success ', 0.05 * (obs[unit_id][4] - next_obs[unit_id][4]))

                if act[unit_id] == 5 and not metrics[unit_id]['if_in_factory']:  ############### transfer ore failed
                    rewards[unit_id] -= 0.1
                    if self.debug:
                        print(unit_id, ' transfer ore failed, power cost ', next_obs[unit_id][2] - obs[unit_id][2])
                              # ' ore cost ', obs[unit_id][4] - next_obs[unit_id][4])

                if obs[unit_id][0] != next_obs[unit_id][0] or obs[unit_id][1] != next_obs[unit_id][1]:  # raward move
                    rewards[unit_id] += 0.1
                    # if self.debug:
                    #     print(unit_id, ' move success ',0.1)

                # if act[unit_id] < 4 and obs[unit_id][0] == next_obs[unit_id][0] \
                #         and obs[unit_id][1] == next_obs[unit_id][1]:  ###################################move failed
                #     rewards[unit_id] -= 0.1
                #     if self.debug:
                #         print(unit_id, ' move failed ', -0.1)

                if obs[unit_id][2] < 500 and act[unit_id] == 6 and metrics[unit_id]['if_in_factory'] \
                        and next_obs[unit_id][2] - obs[unit_id][2] > 0:  ############################### pick up power
                    rewards[unit_id] += 0.001 * (next_obs[unit_id][2] - obs[unit_id][2])
                    if self.debug:
                        print(unit_id, ' pick up power success ', 0.005 * (next_obs[unit_id][2] - obs[unit_id][2]))

                # if act[unit_id] == 8:  ################################################################### do nothing
                #     rewards[unit_id] -= 0.1
                #     if self.debug:
                #         print(unit_id, ' do nothing failed ', -0.1, ' cost power ', next_obs[unit_id][2] - obs[unit_id][2])
        return rewards
