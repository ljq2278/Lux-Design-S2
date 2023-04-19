from typing import Any, Dict
import copy
import gym
import numpy as np
import numpy.typing as npt
from gym import spaces
from lux.config import EnvConfig


class MaObsTransor():
    """
    A simple state based observation to work with in pair with the SimpleUnitDiscreteController

    It contains info only on the first robot, the first factory you own, and some useful features. If there are no owned robots the observation is just zero.
    No information about the opponent is included. This will generate observations for all teams.

    Included features:
    - First robot's stats
    - distance vector to closest ice tile
    - distance vector to first factory

    """

    def __init__(self, env_cfg: EnvConfig) -> None:
        self.env_cfg = env_cfg

        self.pos_dim_start = 0
        self.pos_dim = 2
        self.power_dim_start = self.pos_dim_start + self.pos_dim
        self.power_dim = 1
        self.cargo_dim_start = self.power_dim_start + self.power_dim
        self.cargo_dim = 2
        self.near_rubble_dim_start = self.cargo_dim_start + self.cargo_dim
        self.near_rubble_dim = 5 * 5
        # self.near_robot_dim_start = self.near_rubble_dim_start + self.near_rubble_dim
        # self.near_robot_dim = 5 * 5
        self.min_dist_ices_start = self.near_rubble_dim_start + self.near_rubble_dim  # 30
        self.min_dist_ices = 4 * 2
        self.min_dist_ores_start = self.min_dist_ices_start + self.min_dist_ices  # 38
        self.min_dist_ores = 4 * 2
        self.plyr_ftry_rela_pos_start = self.min_dist_ores_start + self.min_dist_ores  # 46
        self.plyr_ftry_rela_pos = 3 * 2
        self.oppo_ftry_rela_pos_start = self.plyr_ftry_rela_pos_start + self.plyr_ftry_rela_pos  # 52
        self.oppo_ftry_rela_pos = 3 * 2
        self.total_dims = self.oppo_ftry_rela_pos_start + self.oppo_ftry_rela_pos  # 58

        self.observation_space = spaces.Box(-999, 999, shape=(self.total_dims,))
        self.normer = np.array([self.env_cfg.map_size for _ in range(0, 2)] +
                               [self.env_cfg.ROBOTS['HEAVY'].BATTERY_CAPACITY for _ in range(0, 1)] +
                               [self.env_cfg.ROBOTS['HEAVY'].CARGO_SPACE for _ in range(0, 2)] +
                               [self.env_cfg.MAX_RUBBLE for _ in range(0, 25)] +
                               [self.env_cfg.map_size for _ in range(0, 28)])

        self.normer_light = np.array([self.env_cfg.map_size for _ in range(0, 2)] +
                               [self.env_cfg.ROBOTS['HEAVY'].BATTERY_CAPACITY for _ in range(0, 1)] +
                               [self.env_cfg.ROBOTS['HEAVY'].CARGO_SPACE for _ in range(0, 2)] +
                               [self.env_cfg.MAX_RUBBLE for _ in range(0, 25)] +
                               [self.env_cfg.map_size for _ in range(0, 28)])

    # we make this method static so the submission/evaluation code can use this as well
    def sg_to_ma(self, raw_obs: Dict[str, Any]):
        def get_min4(locs, target, map_size):
            ret = [[-1, -1], [-1, -1], [-1, -1], [-1, -1]]
            min_dist2 = [1000000, 1000000, 1000000, 1000000]
            for tp in locs:
                cur_dist2 = (tp[0] - target[0]) ** 2 + (tp[1] - target[1]) ** 2
                if tp[0] < map_size // 2:
                    if tp[1] < map_size // 2:
                        if cur_dist2 < min_dist2[0]:
                            ret[0][0], ret[0][1] = tp[0] - target[0], tp[1] - target[1]
                            min_dist2[0] = cur_dist2
                    else:
                        if cur_dist2 < min_dist2[1]:
                            ret[1][0], ret[1][1] = tp[0] - target[0], tp[1] - target[1]
                            min_dist2[1] = cur_dist2
                else:
                    if tp[1] < map_size // 2:
                        if cur_dist2 < min_dist2[2]:
                            ret[2][0], ret[2][1] = tp[0] - target[0], tp[1] - target[1]
                            min_dist2[2] = cur_dist2
                    else:
                        if cur_dist2 < min_dist2[3]:
                            ret[3][0], ret[3][1] = tp[0] - target[0], tp[1] - target[1]
                            min_dist2[3] = cur_dist2
            return ret[0] + ret[1] + ret[2] + ret[3]

        space_info = np.ones([self.env_cfg.map_size + 4, self.env_cfg.map_size + 4]) * 10
        space_info[2:self.env_cfg.map_size + 2, 2:self.env_cfg.map_size + 2] = raw_obs["board"][
            "rubble"]  # the extent map
        # make the return box
        ret = {}
        # fill the self info
        for player_id, player_info in raw_obs['units'].items():
            ret[player_id] = {}
            for unit_id, unit_info in player_info.items():
                space_info[unit_info['pos'][0] + 2, unit_info['pos'][1] + 2] += 200
                ret[player_id][unit_id] = [-100.0 for _ in range(0, self.total_dims)]
                ret[player_id][unit_id][0], ret[player_id][unit_id][1] = unit_info['pos'][0], unit_info['pos'][
                    1]  # 0,1 pos
                ret[player_id][unit_id][2], ret[player_id][unit_id][3], ret[player_id][unit_id][4] \
                    = unit_info['power'], \
                      unit_info['cargo']['ice'], \
                      unit_info['cargo']['ore']  # 2,3,4 power/ice/ore
        # fill the near space info
        ice_locs = np.argwhere(raw_obs["board"]["ice"] == 1)
        ore_locs = np.argwhere(raw_obs["board"]["ore"] == 1)
        for player_id, player_info in raw_obs['units'].items():
            for unit_id, unit_info in player_info.items():
                near_space = space_info[ret[player_id][unit_id][0] + 2 - 2:ret[player_id][unit_id][0] + 2 + 3,
                             ret[player_id][unit_id][1] + 2 - 2:ret[player_id][unit_id][1] + 2 + 3]
                ret[player_id][unit_id][5:30] = near_space.reshape(-1)
                ret[player_id][unit_id][30:38] = get_min4(ice_locs, unit_info['pos'], self.env_cfg.map_size)
                ret[player_id][unit_id][38:46] = get_min4(ore_locs, unit_info['pos'], self.env_cfg.map_size)
                self_f_count = 0
                oppo_f_count = 0
                # add the factory info
                for p_id, p_info in raw_obs["factories"].items():
                    for f_id, f_info in p_info.items():
                        if p_id == player_id:
                            ret[player_id][unit_id][46 + self_f_count:48 + self_f_count] = \
                                f_info['pos'][0] - unit_info['pos'][0], f_info['pos'][1] - unit_info['pos'][1]
                            self_f_count += 2
                        else:
                            ret[player_id][unit_id][52 + oppo_f_count:54 + oppo_f_count] = \
                                f_info['pos'][0] - unit_info['pos'][0], f_info['pos'][1] - unit_info['pos'][1]
                            oppo_f_count += 2

        return ret, dict([
                (
                    p_id,
                    dict([
                        (
                            u_id,
                            (np.array(u_info) / self.normer).tolist()
                        )
                        for u_id, u_info in p_info.items()
                    ])
                )
                for p_id, p_info in ret.items()
        ])

    def sg_to_ma_light(self, raw_obs: Dict[str, Any]):
        def get_min4(locs, target, map_size):
            ret = [[-1, -1], [-1, -1], [-1, -1], [-1, -1]]
            min_dist2 = [1000000, 1000000, 1000000, 1000000]
            for tp in locs:
                cur_dist2 = (tp[0] - target[0]) ** 2 + (tp[1] - target[1]) ** 2
                if tp[0] < map_size // 2:
                    if tp[1] < map_size // 2:
                        if cur_dist2 < min_dist2[0]:
                            ret[0][0], ret[0][1] = tp[0] - target[0], tp[1] - target[1]
                            min_dist2[0] = cur_dist2
                    else:
                        if cur_dist2 < min_dist2[1]:
                            ret[1][0], ret[1][1] = tp[0] - target[0], tp[1] - target[1]
                            min_dist2[1] = cur_dist2
                else:
                    if tp[1] < map_size // 2:
                        if cur_dist2 < min_dist2[2]:
                            ret[2][0], ret[2][1] = tp[0] - target[0], tp[1] - target[1]
                            min_dist2[2] = cur_dist2
                    else:
                        if cur_dist2 < min_dist2[3]:
                            ret[3][0], ret[3][1] = tp[0] - target[0], tp[1] - target[1]
                            min_dist2[3] = cur_dist2
            return ret[0] + ret[1] + ret[2] + ret[3]

        space_info = np.ones([self.env_cfg.map_size + 4, self.env_cfg.map_size + 4]) * 10
        space_info[2:self.env_cfg.map_size + 2, 2:self.env_cfg.map_size + 2] = raw_obs["board"][
            "rubble"]  # the extent map
        # make the return box
        ret = {}
        # fill the self info
        for player_id, player_info in raw_obs['units'].items():
            ret[player_id] = {}
            for unit_id, unit_info in player_info.items():
                space_info[unit_info['pos'][0] + 2, unit_info['pos'][1] + 2] += 200
                ret[player_id][unit_id] = [-100.0 for _ in range(0, self.total_dims)]
                ret[player_id][unit_id][0], ret[player_id][unit_id][1] = unit_info['pos'][0], unit_info['pos'][
                    1]  # 0,1 pos
                ret[player_id][unit_id][2], ret[player_id][unit_id][3], ret[player_id][unit_id][4] \
                    = unit_info['power'], \
                      unit_info['cargo']['ice'], \
                      unit_info['cargo']['ore']  # 2,3,4 power/ice/ore
        # fill the near space info
        ice_locs = np.argwhere(raw_obs["board"]["ice"] == 1)
        ore_locs = np.argwhere(raw_obs["board"]["ore"] == 1)
        for player_id, player_info in raw_obs['units'].items():
            for unit_id, unit_info in player_info.items():
                near_space = space_info[ret[player_id][unit_id][0] + 2 - 2:ret[player_id][unit_id][0] + 2 + 3,
                             ret[player_id][unit_id][1] + 2 - 2:ret[player_id][unit_id][1] + 2 + 3]
                ret[player_id][unit_id][5:30] = near_space.reshape(-1)
                ret[player_id][unit_id][30:38] = get_min4(ice_locs, unit_info['pos'], self.env_cfg.map_size)
                ret[player_id][unit_id][38:46] = get_min4(ore_locs, unit_info['pos'], self.env_cfg.map_size)
                self_f_count = 0
                oppo_f_count = 0
                # add the factory info
                for p_id, p_info in raw_obs["factories"].items():
                    for f_id, f_info in p_info.items():
                        if p_id == player_id:
                            ret[player_id][unit_id][46 + self_f_count:48 + self_f_count] = \
                                f_info['pos'][0] - unit_info['pos'][0], f_info['pos'][1] - unit_info['pos'][1]
                            self_f_count += 2
                        else:
                            ret[player_id][unit_id][52 + oppo_f_count:54 + oppo_f_count] = \
                                f_info['pos'][0] - unit_info['pos'][0], f_info['pos'][1] - unit_info['pos'][1]
                            oppo_f_count += 2

        return ret, dict([
                (
                    p_id,
                    dict([
                        (
                            u_id,
                            (np.array(u_info) / self.normer_light).tolist()
                        )
                        for u_id, u_info in p_info.items()
                    ])
                )
                for p_id, p_info in ret.items()
        ])
