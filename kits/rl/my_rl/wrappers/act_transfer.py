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

    def u_choice_to_raw(self, ind):
        if self.action_space_unit.u_move_dim_start <= ind < self.action_space_unit.u_move_dim_start + self.action_space_unit.u_move_dim:
            return np.array([0, ind + 1, 0, 0, 0, 1])
        elif self.action_space_unit.u_pickup_dim_start <= ind < self.action_space_unit.u_pickup_dim_start + self.action_space_unit.u_pickup_dim:
            for i in range(0, self.action_space_unit.u_pickup_dim):
                if ind == self.action_space_unit.u_pickup_dim_start + i:
                    return np.array([2, 0, 4, 100 + ((i ** 2) * 200), 0, 1])
        elif self.action_space_unit.u_transfer_dim_start <= ind < self.action_space_unit.u_transfer_dim_start + self.action_space_unit.u_transfer_dim:
            for i in range(0, self.action_space_unit.u_transfer_dim):
                if i == 0:
                    return np.array([1, 0, 0, self.env_cfg.max_transfer_amount, 0, 1])
                else:
                    return np.array([1, 0, 1, self.env_cfg.max_transfer_amount, 0, 1])
        elif self.action_space_unit.u_dig_dim_start == ind:
            return np.array([3, 0, 0, 0, 0, 1])
        else:
            return None

    def wrap_to_raw(self, f_actions, u_actions, f_dict, u_dict):
        raw_action = dict()
        for f_id, f_info in f_dict.items():
            choice = f_actions[f_info.pos.x][f_info.pos.y]
            if choice != self.action_space_factory.f_noop_dim_start:
                raw_action[f_id] = choice
        for u_id, u_info in u_dict.items():
            choice = u_actions[u_info.pos.x][u_info.pos.y]
            u_raw_act = self.u_choice_to_raw(choice)
            if u_raw_act is not None:
                raw_action[u_id] = u_raw_act
        return raw_action
