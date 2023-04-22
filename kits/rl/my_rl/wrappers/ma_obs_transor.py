from typing import Any, Dict
import copy
import gym
import numpy as np
import numpy.typing as npt
from gym import spaces
from lux.config import EnvConfig
from wrappers.obs_space import ObsSpace


class MaObsTransor(ObsSpace):
    """
    A simple state based observation to work with in pair with the SimpleUnitDiscreteController

    It contains info only on the first robot, the first factory you own, and some useful features. If there are no owned robots the observation is just zero.
    No information about the opponent is included. This will generate observations for all teams.

    Included features:
    - First robot's stats
    - distance vector to closest ice tile
    - distance vector to first factory

    """

    def __init__(self, env, env_cfg: EnvConfig) -> None:
        super().__init__(env_cfg)
        self.env = env
        self.observation_space = spaces.Box(-999, 999, shape=(self.total_dims,))

    def _get_min_rela_pos(self, pos, f_pos_list, extra_list=None):
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
        if extra_list is None:
            return min_rela_pos
        else:
            return min_rela_pos, extra_list[min_ind]

    # we make this method static so the submission/evaluation code can use this as well
    def sg_to_ma(self, raw_obs: Dict[str, Any]):
        player_set = set(raw_obs["factories"].keys())
        space_info = np.ones([self.env_cfg.map_size + 4, self.env_cfg.map_size + 4]) * self.env_cfg.MAX_RUBBLE
        space_info[2:self.env_cfg.map_size + 2, 2:self.env_cfg.map_size + 2] = raw_obs["board"]["rubble"]  # rubble map
        ice_locs = np.argwhere(raw_obs["board"]["ice"] == 1)
        ore_locs = np.argwhere(raw_obs["board"]["ore"] == 1)
        ################################################################### fill the robot map
        for player_id, player_info in raw_obs['units'].items():
            for unit_id, unit_info in player_info.items():
                space_info[
                    unit_info['pos'][0] + 2, unit_info['pos'][1] + 2] += self.env_cfg.MAX_RUBBLE  # complete space

        ############################################################# get factory pos
        players_f_dict = {}
        players_f_water_dict = {}
        oppo_players_f_dict = {}
        for p_id, p_info in raw_obs["factories"].items():
            players_f_dict[p_id] = []
            players_f_water_dict[p_id] = []
            for i, (f_id, f_info) in enumerate(p_info.items()):
                players_f_dict[p_id].append(f_info['pos'])
                players_f_water_dict[p_id].append(f_info['cargo']['water'])
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

                f_min_rela_pos, f_water = self._get_min_rela_pos(
                    unit_info['pos'], players_f_dict[player_id], players_f_water_dict[player_id])
                f_oppo_min_rela_pos = self._get_min_rela_pos(unit_info['pos'], oppo_players_f_dict[player_id])
                ice_min_rela_pos = self._get_min_rela_pos(unit_info['pos'], ice_locs)
                ore_min_rela_pos = self._get_min_rela_pos(unit_info['pos'], ore_locs)
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
                # near_space
                ret[player_id][unit_id][self.near_space_start: self.near_space_start + self.near_space] = near_space

                # add the near ice info
                ret[player_id][unit_id][
                self.nearest_ice_pos_start:self.nearest_ice_pos_start + self.nearest_ice_pos
                ] = ice_min_rela_pos

                # add the near ore info
                ret[player_id][unit_id][
                self.nearest_ore_pos_start:self.nearest_ore_pos_start + self.nearest_ore_pos
                ] = ore_min_rela_pos

                # add near factory
                ret[player_id][unit_id][
                self.nearest_factory_pos_start:self.nearest_factory_pos_start + self.nearest_factory_pos
                ] = f_min_rela_pos

                # add near oppo factory
                ret[player_id][unit_id][
                self.nearest_oppo_factory_pos_start:self.nearest_oppo_factory_pos_start + self.nearest_oppo_factory_pos
                ] = f_oppo_min_rela_pos

                # add near water
                ret[player_id][unit_id][self.nearest_factory_water_start] = f_water

                # is day ?
                # ret[player_id][unit_id][-1] = int(raw_obs['real_env_steps'] % self.env_cfg.CYCLE_LENGTH < self.env_cfg.DAY_LENGTH)
                # the time
                ret[player_id][unit_id][self.day_or_night_start] = \
                    (raw_obs['real_env_steps'] % self.env_cfg.DAY_LENGTH) / self.env_cfg.DAY_LENGTH

                ret[player_id][unit_id][self.has_ice_start] = 1 if unit_info['cargo']['ice'] > 0 else 0
                ret[player_id][unit_id][self.has_ore_start] = 1 if unit_info['cargo']['ore'] > 0 else 0

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
