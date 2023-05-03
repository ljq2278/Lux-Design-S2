from typing import Any, Dict
import copy
import gym
import numpy as np
import numpy.typing as npt
from gym import spaces
from lux.config import EnvConfig
from wrappers.obs_space_levels import ObsSpaceUnit, ObsSpaceFactory


class MaObsTransorUnit(ObsSpaceUnit):

    def __init__(self, env, env_cfg: EnvConfig, if_mask=False) -> None:
        super().__init__(env_cfg)
        self.env = env
        self.if_mask = if_mask
        self.observation_space = spaces.Box(-999, 999, shape=(self.total_dims,))

    def _get_min_rela_pos(self, pos, f_pos_list):
        min_dist2 = 1000000000
        min_rela_pos = [self.env_cfg.map_size * 5, self.env_cfg.map_size * 5]
        min_ind = None
        for i, f_pos in enumerate(f_pos_list):
            cur_dist2 = (f_pos[0] - pos[0]) ** 2 + (f_pos[1] - pos[1]) ** 2
            if cur_dist2 < min_dist2:
                min_rela_pos = [f_pos[0] - pos[0], f_pos[1] - pos[1]]
                min_dist2 = cur_dist2
                min_ind = i
            elif cur_dist2 == min_dist2:
                if np.random.random() > 0.5:
                    min_rela_pos = [f_pos[0] - pos[0], f_pos[1] - pos[1]]
                    min_ind = i
        return min_rela_pos, min_ind

    # we make this method static so the submission/evaluation code can use this as well
    def sg_to_ma(self, raw_obs: Dict[str, Any], target_factory_pos, target_task):
        player_set = set(raw_obs["factories"].keys())
        space_info = np.ones([self.env_cfg.map_size + 4, self.env_cfg.map_size + 4]) * self.env_cfg.MAX_RUBBLE
        space_info[2:self.env_cfg.map_size + 2, 2:self.env_cfg.map_size + 2] = raw_obs["board"]["rubble"]  # rubble map
        ice_locs = np.argwhere(raw_obs["board"]["ice"] == 1)
        ore_locs = np.argwhere(raw_obs["board"]["ore"] == 1)
        robot_rubble_identied = self.env_cfg.MAX_RUBBLE * 2
        ################################################################### fill the robot map
        for player_id, player_info in raw_obs['units'].items():
            for unit_id, unit_info in player_info.items():
                space_info[
                    unit_info['pos'][0] + 2, unit_info['pos'][1] + 2] += robot_rubble_identied  # complete space

        ############################################################# get factory pos
        players_f_dict = {}
        players_f_water_dict = {}
        players_f_metal_dict = {}
        players_f_ice_dict = {}
        players_f_ore_dict = {}
        oppo_players_f_dict = {}
        for p_id, p_info in raw_obs["factories"].items():
            players_f_dict[p_id] = []
            players_f_water_dict[p_id] = []
            players_f_metal_dict[p_id] = []
            players_f_ice_dict[p_id] = []
            players_f_ore_dict[p_id] = []
            for i, (f_id, f_info) in enumerate(p_info.items()):
                players_f_dict[p_id].append(f_info['pos'])
                players_f_water_dict[p_id].append(f_info['cargo']['water'])
                players_f_metal_dict[p_id].append(f_info['cargo']['metal'])
                players_f_ice_dict[p_id].append(f_info['cargo']['ice'])
                players_f_ore_dict[p_id].append(f_info['cargo']['ore'])
                for pid in player_set:
                    if pid != p_id:
                        if pid not in oppo_players_f_dict.keys():
                            oppo_players_f_dict[pid] = []
                        oppo_players_f_dict[pid].append(f_info['pos'])
        #########################################################################
        ret = {}
        for player_id, player_info in raw_obs['units'].items():
            ret[player_id] = {}
            for unit_id, unit_info in player_info.items():
                ret[player_id][unit_id] = [-100 for _ in range(0, self.total_dims)]
                if len(players_f_dict['player_0']) == 0 or len(players_f_dict['player_1']) == 0:
                    continue
                #################################################### get near f_pos oppo_f_pos ice ore

                f_oppo_min_rela_pos, _ = self._get_min_rela_pos(unit_info['pos'], oppo_players_f_dict[player_id])
                if target_task == 'water':
                    target_min_rela_pos, _ = self._get_min_rela_pos(unit_info['pos'], ice_locs)
                elif target_task == 'metal':
                    target_min_rela_pos, _ = self._get_min_rela_pos(unit_info['pos'], ore_locs)
                else:
                    target_min_rela_pos = -1  # dig for plant, no implement
                ############################################################## near space

                near_space = space_info[
                             unit_info['pos'][0] + 2 - 2:unit_info['pos'][0] + 2 + 3,
                             unit_info['pos'][1] + 2 - 2:unit_info['pos'][1] + 2 + 3
                             ].reshape(-1)
                ############################################################# start build obs feature

                # 0,1 pos
                ret[player_id][unit_id][0:2] = unit_info['pos']
                # 2,3,4 power/ice/ore
                ret[player_id][unit_id][2:5] = [unit_info['power'], unit_info['cargo']['ice'],
                                                unit_info['cargo']['ore']]
                # near_space, the centr tile have the self robot
                ret[player_id][unit_id][self.near_space_start: self.near_space_start + self.near_space] = near_space
                ret[player_id][unit_id][int((self.near_space_start + self.near_space_start + self.near_space) // 2)] \
                    -= robot_rubble_identied

                # add target factory
                ret[player_id][unit_id][self.target_factory_pos_start:self.target_factory_pos_start + self.target_factory_pos] \
                    = target_factory_pos[0] - unit_info['pos'][0], target_factory_pos[1] - unit_info['pos'][1]

                # add target tile
                ret[player_id][unit_id][self.target_pos_start:self.target_pos_start + self.target_pos] = target_min_rela_pos

                # add near oppo factory
                ret[player_id][unit_id][self.nearest_oppo_factory_pos_start:self.nearest_oppo_factory_pos_start + self.nearest_oppo_factory_pos] = f_oppo_min_rela_pos

                # add time info
                ret[player_id][unit_id][self.day_or_night_start] = \
                    (raw_obs['real_env_steps'] % self.env_cfg.DAY_LENGTH) / self.env_cfg.DAY_LENGTH

        if self.if_mask:
            return ret, dict([
                (
                    p_id,
                    dict([
                        (
                            u_id,
                            (np.array(u_info) / self.normer * self.mask).tolist()
                        )
                        for u_id, u_info in p_info.items()
                    ])
                )
                for p_id, p_info in ret.items()
            ])
        else:
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


class MaObsTransorFactory(ObsSpaceFactory):

    def __init__(self, env, env_cfg: EnvConfig, if_mask=False) -> None:
        super().__init__(env_cfg)
        self.env = env
        self.if_mask = if_mask
        self.observation_space = spaces.Box(-999, 999, shape=(self.total_dims,))

    # we make this method static so the submission/evaluation code can use this as well
    def sg_to_ma(self, raw_obs: Dict[str, Any]):
        player_set = set(raw_obs["factories"].keys())
        space_info = np.ones([self.env_cfg.map_size + 4, self.env_cfg.map_size + 4]) * self.env_cfg.MAX_RUBBLE
        space_info[2:self.env_cfg.map_size + 2, 2:self.env_cfg.map_size + 2] = raw_obs["board"]["rubble"]  # rubble map
        ice_locs = np.argwhere(raw_obs["board"]["ice"] == 1)
        ore_locs = np.argwhere(raw_obs["board"]["ore"] == 1)
        robot_rubble_identied = self.env_cfg.MAX_RUBBLE * 2
        ################################################################### fill the robot map
        for player_id, player_info in raw_obs['units'].items():
            for unit_id, unit_info in player_info.items():
                space_info[
                    unit_info['pos'][0] + 2, unit_info['pos'][1] + 2] += robot_rubble_identied  # complete space

        ############################################################# get factory pos
        players_f_dict = {}
        players_f_water_dict = {}
        players_f_metal_dict = {}
        players_f_ice_dict = {}
        players_f_ore_dict = {}
        oppo_players_f_dict = {}
        for p_id, p_info in raw_obs["factories"].items():
            players_f_dict[p_id] = []
            players_f_water_dict[p_id] = []
            players_f_metal_dict[p_id] = []
            players_f_ice_dict[p_id] = []
            players_f_ore_dict[p_id] = []
            for i, (f_id, f_info) in enumerate(p_info.items()):
                players_f_dict[p_id].append(f_info['pos'])
                players_f_water_dict[p_id].append(f_info['cargo']['water'])
                players_f_metal_dict[p_id].append(f_info['cargo']['metal'])
                players_f_ice_dict[p_id].append(f_info['cargo']['ice'])
                players_f_ore_dict[p_id].append(f_info['cargo']['ore'])
                for pid in player_set:
                    if pid != p_id:
                        if pid not in oppo_players_f_dict.keys():
                            oppo_players_f_dict[pid] = []
                        oppo_players_f_dict[pid].append(f_info['pos'])
        #########################################################################
        ret = {}
        for player_id, player_info in raw_obs['units'].items():
            ret[player_id] = {}
            for unit_id, unit_info in player_info.items():
                ret[player_id][unit_id] = [-100 for _ in range(0, self.total_dims)]
                if len(players_f_dict['player_0']) == 0 or len(players_f_dict['player_1']) == 0:
                    continue
                #################################################### get near f_pos oppo_f_pos ice ore

                f_oppo_min_rela_pos, _ = self._get_min_rela_pos(unit_info['pos'], oppo_players_f_dict[player_id])
                if target_task == 'water':
                    target_min_rela_pos, _ = self._get_min_rela_pos(unit_info['pos'], ice_locs)
                elif target_task == 'metal':
                    target_min_rela_pos, _ = self._get_min_rela_pos(unit_info['pos'], ore_locs)
                else:
                    target_min_rela_pos = -1  # dig for plant, no implement
                ############################################################## near space

                near_space = space_info[
                             unit_info['pos'][0] + 2 - 2:unit_info['pos'][0] + 2 + 3,
                             unit_info['pos'][1] + 2 - 2:unit_info['pos'][1] + 2 + 3
                             ].reshape(-1)
                ############################################################# start build obs feature

                # 0,1 pos
                ret[player_id][unit_id][0:2] = unit_info['pos']
                # 2,3,4 power/ice/ore
                ret[player_id][unit_id][2:5] = [unit_info['power'], unit_info['cargo']['ice'],
                                                unit_info['cargo']['ore']]
                # near_space, the centr tile have the self robot
                ret[player_id][unit_id][self.near_space_start: self.near_space_start + self.near_space] = near_space
                ret[player_id][unit_id][int((self.near_space_start + self.near_space_start + self.near_space) // 2)] \
                    -= robot_rubble_identied

                # add target factory
                ret[player_id][unit_id][self.target_factory_pos_start:self.target_factory_pos_start + self.target_factory_pos] \
                    = target_factory_pos[0] - unit_info['pos'][0], target_factory_pos[1] - unit_info['pos'][1]

                # add target tile
                ret[player_id][unit_id][self.target_pos_start:self.target_pos_start + self.target_pos] = target_min_rela_pos

                # add near oppo factory
                ret[player_id][unit_id][self.nearest_oppo_factory_pos_start:self.nearest_oppo_factory_pos_start + self.nearest_oppo_factory_pos] = f_oppo_min_rela_pos

                # add time info
                ret[player_id][unit_id][self.day_or_night_start] = \
                    (raw_obs['real_env_steps'] % self.env_cfg.DAY_LENGTH) / self.env_cfg.DAY_LENGTH

        if self.if_mask:
            return ret, dict([
                (
                    p_id,
                    dict([
                        (
                            u_id,
                            (np.array(u_info) / self.normer * self.mask).tolist()
                        )
                        for u_id, u_info in p_info.items()
                    ])
                )
                for p_id, p_info in ret.items()
            ])
        else:
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
