import sys
from typing import Any, Dict

import numpy as np
import numpy.typing as npt
from gym import spaces

from wrappers.controllers import Controller

class SimpleUnitDiscreteController_multiagent(Controller):
    def __init__(self, env_cfg) -> None:
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
        self.env_cfg = env_cfg
        self.move_act_dims = 4
        self.transfer_ice_act_dims = 1
        self.transfer_ore_act_dims = 1
        self.pickup_act_dims = 1
        self.dig_act_dims = 1
        self.no_op_dims = 1

        self.move_dim_high = self.move_act_dims
        self.transfer_ice_dim_high = self.move_dim_high + self.transfer_ice_act_dims
        self.transfer_ore_dim_high = self.transfer_ice_dim_high + self.transfer_ore_act_dims
        self.pickup_dim_high = self.transfer_ore_dim_high + self.pickup_act_dims
        self.dig_dim_high = self.pickup_dim_high + self.dig_act_dims
        self.no_op_dim_high = self.dig_dim_high + self.no_op_dims

        self.total_act_dims = self.no_op_dim_high
        action_space = spaces.Discrete(self.total_act_dims)
        super().__init__(action_space)

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
        if water>150:
            return True
        return False

    def _can_build(self, power, metal):
        if power>500 and metal>100:
            return True
        return False

    def action_to_lux_action(
        self, agent: str, obs: Dict[str, Any], actions: Dict[str, npt.NDArray]
    ):
        shared_obs = obs["player_0"]
        lux_action = dict()
        units = shared_obs["units"][agent]
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

            # simple trick to help agents conserve power is to avoid updating the action queue
            # if the agent was previously trying to do that particular action already
            if len(unit["action_queue"]) > 0 and len(action_queue) > 0:
                same_actions = (unit["action_queue"][0] == action_queue[0]).all()
                if same_actions:
                    no_op = True
            if not no_op:
                lux_action[unit_id] = action_queue

        factories = shared_obs["factories"][agent]

        # build a single heavy
        for f_id in factories.keys():
            if self._can_build(factories[f_id]['power'],factories[f_id]['cargo']['metal']):
                lux_action[f_id] = 1
            else:
                if self._can_water(factories[f_id]['water']):
                    lux_action[f_id] = 2

        return lux_action

    # def action_masks(self, agent: str, obs: Dict[str, Any]):
    #     """
    #     Defines a simplified action mask for this controller's action space
    #
    #     Doesn't account for whether robot has enough power
    #     """
    #
    #     # compute a factory occupancy map that will be useful for checking if a board tile
    #     # has a factory and which team's factory it is.
    #     shared_obs = obs[agent]
    #     factory_occupancy_map = (
    #         np.ones_like(shared_obs["board"]["rubble"], dtype=int) * -1
    #     )
    #     factories = dict()
    #     for player in shared_obs["factories"]:
    #         factories[player] = dict()
    #         for unit_id in shared_obs["factories"][player]:
    #             f_data = shared_obs["factories"][player][unit_id]
    #             f_pos = f_data["pos"]
    #             # store in a 3x3 space around the factory position it's strain id.
    #             factory_occupancy_map[
    #                 f_pos[0] - 1 : f_pos[0] + 2, f_pos[1] - 1 : f_pos[1] + 2
    #             ] = f_data["strain_id"]
    #
    #     units = shared_obs["units"][agent]
    #     action_mask = np.zeros((self.total_act_dims), dtype=bool)
    #     for unit_id in units.keys():
    #         action_mask = np.zeros(self.total_act_dims)
    #         # movement is always valid
    #         action_mask[:4] = True
    #
    #         # transferring is valid only if the target exists
    #         unit = units[unit_id]
    #         pos = np.array(unit["pos"])
    #         # a[1] = direction (0 = center, 1 = up, 2 = right, 3 = down, 4 = left)
    #         move_deltas = np.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]])
    #         for i, move_delta in enumerate(move_deltas):
    #             transfer_pos = np.array(
    #                 [pos[0] + move_delta[0], pos[1] + move_delta[1]]
    #             )
    #             # check if theres a factory tile there
    #             if (
    #                 transfer_pos[0] < 0
    #                 or transfer_pos[1] < 0
    #                 or transfer_pos[0] >= len(factory_occupancy_map)
    #                 or transfer_pos[1] >= len(factory_occupancy_map[0])
    #             ):
    #                 continue
    #             factory_there = factory_occupancy_map[transfer_pos[0], transfer_pos[1]]
    #             if factory_there in shared_obs["teams"][agent]["factory_strains"]:
    #                 action_mask[
    #                     self.transfer_dim_high - self.transfer_act_dims + i
    #                 ] = True
    #
    #         factory_there = factory_occupancy_map[pos[0], pos[1]]
    #         on_top_of_factory = (
    #             factory_there in shared_obs["teams"][agent]["factory_strains"]
    #         )
    #
    #         # dig is valid only if on top of tile with rubble or resources or lichen
    #         board_sum = (
    #             shared_obs["board"]["ice"][pos[0], pos[1]]
    #             + shared_obs["board"]["ore"][pos[0], pos[1]]
    #             + shared_obs["board"]["rubble"][pos[0], pos[1]]
    #             + shared_obs["board"]["lichen"][pos[0], pos[1]]
    #         )
    #         if board_sum > 0 and not on_top_of_factory:
    #             action_mask[
    #                 self.dig_dim_high - self.dig_act_dims : self.dig_dim_high
    #             ] = True
    #
    #         # pickup is valid only if on top of factory tile
    #         if on_top_of_factory:
    #             action_mask[
    #                 self.pickup_dim_high - self.pickup_act_dims : self.pickup_dim_high
    #             ] = True
    #             action_mask[
    #                 self.dig_dim_high - self.dig_act_dims : self.dig_dim_high
    #             ] = False
    #
    #         # no-op is always valid
    #         action_mask[-1] = True
    #         break
    #     return action_mask
