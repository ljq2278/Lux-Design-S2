import sys
from typing import Any, Dict

import numpy as np
import numpy.typing as npt
from gym import spaces
from lux.config import EnvConfig
from wrappers.act_space_levels import ActSpaceUnit, ActSpaceFactory
from wrappers.obs_space_levels import ObsSpaceUnit


class MaActTransorUnit(ActSpaceUnit):
    def __init__(self, env, env_cfg: EnvConfig) -> None:
        super().__init__(env_cfg)
        self.env = env
        self.action_space = spaces.Discrete(self.total_act_dims)

    def _is_move_action(self, id):
        return id < self.move_dim_high

    def _can_move(self, id, obs):
        t_rubble = ObsSpaceUnit.move_target_rubble(obs, id) % 200
        if obs[ObsSpaceUnit.power_dim_start] < 20 + 1 * t_rubble + 10:
            return False
        return True

    def _get_move_action(self, id):
        # move direction is id + 1 since we don't allow move center here
        return np.array([0, id + 1, 0, 0, 0, 1])

    def _is_transfer_target_action(self, id):
        return id < self.transfer_target_dim_high

    def _get_transfer_target_action(self, task_type):
        if task_type == 'ice':
            return np.array([1, 0, 0, self.env_cfg.max_transfer_amount, 0, 1])
        elif task_type == 'ore':
            return np.array([1, 0, 1, self.env_cfg.max_transfer_amount, 0, 1])
        else:
            if np.random.random() < 0.5:
                return np.array([1, 0, 0, self.env_cfg.max_transfer_amount, 0, 1])
            else:
                return np.array([1, 0, 1, self.env_cfg.max_transfer_amount, 0, 1])

    def _is_pickup_action(self, id):
        return id < self.pickup_dim_high

    def _get_pickup_action(self, id):
        return np.array([2, 0, 4, self.env_cfg.max_transfer_amount // 5, 0, 1])

    def _is_dig_action(self, id):
        return id < self.dig_dim_high

    def _can_dig(self, obs):
        if obs[ObsSpaceUnit.power_dim_start] < 60 + 10:
            return False
        return True

    def _get_dig_action(self, id):
        return np.array([3, 0, 0, 0, 0, 1])

    def ma_to_sg(self, actions: Dict[str, npt.NDArray], obs_unit):

        raw_action = dict()
        # units = obs_unit.keys()
        for unit_id, choice in actions.items():
            # unit = units[unit_id]
            action_queue = []
            no_op = False
            if self._is_move_action(choice):
                if self._can_move(choice, obs_unit[unit_id]):
                    action_queue = [self._get_move_action(choice)]
                else:
                    no_op = True
            elif self._is_transfer_target_action(choice):
                action_queue = [self._get_transfer_target_action(ObsSpaceUnit.int_to_task_type(obs_unit[unit_id][ObsSpaceUnit.task_type_start]))]
            elif self._is_pickup_action(choice):
                action_queue = [self._get_pickup_action(choice)]
            elif self._is_dig_action(choice):
                if self._can_dig(obs_unit[unit_id]):
                    action_queue = [self._get_dig_action(choice)]
                else:
                    no_op = True
            else:
                # action is a no_op, so we don't update the action queue
                no_op = True
            # if len(unit["action_queue"]) > 0 and len(action_queue) > 0:
            #     same_actions = (unit["action_queue"][0] == action_queue[0]).all()
            #     if same_actions:
            #         no_op = True
            if not no_op:
                raw_action[unit_id] = action_queue

        return raw_action


class MaActTransorFactory(ActSpaceFactory):
    def __init__(self, env, env_cfg: EnvConfig) -> None:
        super().__init__(env_cfg)
        self.env = env
        self.action_space = spaces.Discrete(self.total_act_dims)

    def _is_build_light_action(self, id):
        return id < self.build_light_high

    def _is_build_heavy_action(self, id):
        return id < self.build_heavy_high

    def _is_water_lichen_action(self, id):
        return id < self.water_lichen_high

    def _can_water(self, water):
        if water >= 150:
            return True
        return False

    def _can_build(self, power, metal):
        if power >= 500 and metal >= 100:
            return True
        return False

    def ma_to_sg(self, actions: Dict[str, npt.NDArray], obs_info, player):
        raw_action = dict()
        for f_id, choice in actions.items():
            if self._is_build_light_action(choice):
                raw_action[f_id] = 0
            elif self._is_build_heavy_action(choice):
                raw_action[f_id] = 1
            elif self._is_water_lichen_action(choice):
                raw_action[f_id] = 2
        return raw_action
