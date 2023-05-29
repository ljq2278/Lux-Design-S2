import sys
from typing import Any, Dict

import numpy as np
import numpy.typing as npt
from gym import spaces
from lux.config import EnvConfig
from wrappers.act_space_conv import ActSpaceFactory, ActSpaceUnit


class ActTransfer:
    def __init__(self, env, env_cfg: EnvConfig) -> None:
        self.env = env
        self.env_cfg = env_cfg
        self.action_space_factory = ActSpaceFactory()
        self.action_space_unit = ActSpaceUnit()

    def f_choice_to_raw(self, ind):
        pass

    def u_choice_to_raw(self, ind, u_nearest_f_dist, u_pos):
        if self.action_space_unit.u_move_dim_start <= ind < self.action_space_unit.u_move_dim_start + self.action_space_unit.u_move_dim:
            return [np.array([0, ind + 1, 0, 0, 0, 1])]
        elif self.action_space_unit.u_pickup_dim_start <= ind < self.action_space_unit.u_pickup_dim_start + self.action_space_unit.u_pickup_dim:
            return [np.array([2, 0, 4, 100 + (((ind - self.action_space_unit.u_pickup_dim_start) ** 2) * 500), 0, 1])]
        elif self.action_space_unit.u_transfer_dim_start <= ind < self.action_space_unit.u_transfer_dim_start + self.action_space_unit.u_transfer_dim:
            dist_x, dist_y = [int(t) for t in u_nearest_f_dist.split('_')]
            if dist_x == 0 and dist_y == 0:
                dir = 0
            elif abs(dist_x) > abs(dist_y):
                dir = 2 if dist_x > 0 else 4
            else:
                dir = 3 if dist_y > 0 else 1
            if abs(dist_x) + abs(dist_y) <= 3 and max(abs(dist_x), abs(dist_y)) <= 2:
                # print('u_transfer: ', [np.array([1, dir, int(np.random.random() > 0.5), self.env_cfg.max_transfer_amount, 0, 1])], dist_x, dist_y, u_pos)
                return [np.array([1, dir, int(np.random.random() > 0.5), self.env_cfg.max_transfer_amount, 0, 1])]
            else:
                # print('u_transfer: ', [np.array([0, dir, 0, 0, 0, 1])], dist_x, dist_y, u_pos)
                return [np.array([0, dir, 0, 0, 0, 1])]
            # return [np.array([1, (ind - self.action_space_unit.u_transfer_dim_start) % (self.action_space_unit.u_transfer_dim // 2),
            #                   (ind - self.action_space_unit.u_transfer_dim_start) // (self.action_space_unit.u_transfer_dim // 2), self.env_cfg.max_transfer_amount, 0, 1])]
        elif self.action_space_unit.u_dig_dim_start == ind:
            return [np.array([3, 0, 0, 0, 0, 1])]
        else:
            return None

    def wrap_to_raw(self, f_actions, u_actions, f_dict, u_dict):
        raw_action = dict()
        u_nearest_f = dict()
        for u_id, u_info in u_dict.items():
            for f_id, f_info in f_dict.items():
                if str(u_info.pos.x) + '_' + str(u_info.pos.y) not in u_nearest_f.keys():
                    u_nearest_f[str(u_info.pos.x) + '_' + str(u_info.pos.y)] = str(f_info.pos.x - u_info.pos.x) + '_' + str(f_info.pos.y - u_info.pos.y)
                else:
                    ori_dist_x, ori_dist_y = u_nearest_f[str(u_info.pos.x) + '_' + str(u_info.pos.y)].split('_')
                    if abs(f_info.pos.x - u_info.pos.x) + abs(f_info.pos.y - u_info.pos.y) < abs(int(ori_dist_x)) + abs(int(ori_dist_y)):
                        u_nearest_f[str(u_info.pos.x) + '_' + str(u_info.pos.y)] = str(f_info.pos.x - u_info.pos.x) + '_' + str(f_info.pos.y - u_info.pos.y)
        for f_id, f_info in f_dict.items():
            choice = f_actions[f_info.pos.x][f_info.pos.y]
            if choice != self.action_space_factory.f_noop_dim_start:
                raw_action[f_id] = choice
        for u_id, u_info in u_dict.items():
            choice = u_actions[u_info.pos.x][u_info.pos.y]
            u_raw_act = self.u_choice_to_raw(choice, u_nearest_f[str(u_info.pos.x) + '_' + str(u_info.pos.y)], str(u_info.pos.x) + '_' + str(u_info.pos.y))
            if u_raw_act is not None:
                raw_action[u_id] = u_raw_act
        return raw_action
