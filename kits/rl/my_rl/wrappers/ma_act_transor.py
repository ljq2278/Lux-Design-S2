import sys
from typing import Any, Dict

import numpy as np
import numpy.typing as npt
from gym import spaces
from lux.config import EnvConfig
from wrappers.act_space import ActSpace

class MaActTransor(ActSpace):
    def __init__(self, env, env_cfg: EnvConfig) -> None:
        """
        A simple controller that controls only the robot that will get spawned.
        Moreover, it will always try to spawn one heavy robot if there are none regardless of action given

        For the robot unit
        - 4 cardinal direction movement (4 dims)
        - a move center no-op action (1 dim)
        - transfer action just for transferring ice in 4 cardinal directions or center (5)
        - pickup action for power (1 dims)
        - dig action (1 dim)
        - no op action (1 dim) - equivalent to not submitting an action queue which costs power

        It does not include
        - self destruct action
        - recharge action
        - planning (via actions executing multiple times or repeating actions)
        - factory actions
        - transferring power or resources other than ice

        To help understand how to this controller works to map one action space to the original lux action space,
        see how the lux action space is defined in luxai_s2/spaces/action.py

        """
        super().__init__(env_cfg)
        self.env = env
        self.action_space = spaces.Discrete(self.total_act_dims)

    def _is_move_action(self, id):
        return id < self.move_dim_high

    def _get_move_action(self, id):
        # move direction is id + 1 since we don't allow move center here
        return np.array([0, id + 1, 0, 0, 0, 1])

    def _is_transfer_ice_action(self, id):
        return id < self.transfer_ice_dim_high

    def _is_transfer_ore_action(self, id):
        return id < self.transfer_ore_dim_high

    def _get_transfer_ice_action(self, id):
        return np.array([1, 0, 0, self.env_cfg.max_transfer_amount, 0, 1])

    def _get_transfer_ore_action(self, id):
        return np.array([1, 0, 1, self.env_cfg.max_transfer_amount, 0, 1])

    def _is_pickup_action(self, id):
        return id < self.pickup_dim_high

    def _get_pickup_action(self, id):
        return np.array([2, 0, 4, self.env_cfg.max_transfer_amount, 0, 1])

    def _is_dig_action(self, id):
        return id < self.dig_dim_high

    def _get_dig_action(self, id):
        return np.array([3, 0, 0, 0, 0, 1])

    def _can_water(self, water):
        if water > 150:
            return True
        return False

    def _can_build(self, power, metal):
        if power > 500 and metal > 100:
            return True
        return False

    def _can_build_light(self, power, metal):
        if power > 50 and metal > 10:
            return True
        return False

    def ma_to_sg(self, actions: Dict[str, npt.NDArray], obs_info, player):

        raw_action = dict()
        units = obs_info["units"][player]
        for unit_id, choice in actions.items():
            unit = units[unit_id]
            # choice = action
            action_queue = []
            no_op = False
            if self._is_move_action(choice):
                action_queue = [self._get_move_action(choice)]
            elif self._is_transfer_ice_action(choice):
                action_queue = [self._get_transfer_ice_action(choice)]
            elif self._is_transfer_ore_action(choice):
                action_queue = [self._get_transfer_ore_action(choice)]
            elif self._is_pickup_action(choice):
                action_queue = [self._get_pickup_action(choice)]
            elif self._is_dig_action(choice):
                action_queue = [self._get_dig_action(choice)]
            else:
                # action is a no_op, so we don't update the action queue
                no_op = True
            if len(unit["action_queue"]) > 0 and len(action_queue) > 0:
                same_actions = (unit["action_queue"][0] == action_queue[0]).all()
                if same_actions:
                    no_op = True
            if not no_op:
                raw_action[unit_id] = action_queue

        factories = obs_info["factories"][player]

        # build a single heavy
        for f_id in factories.keys():
            if self._can_build(factories[f_id]['power'], factories[f_id]['cargo']['metal']):
                raw_action[f_id] = 1
                if obs_info["real_env_steps"] > 5:
                    print('########################### build a HEAVY')
            else:
                if self._can_water(factories[f_id]['cargo']['water']):
                    # print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ can water ')
                    raw_action[f_id] = 2

        return raw_action
