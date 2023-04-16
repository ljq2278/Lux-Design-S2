from typing import Any, Dict

import gym
import numpy as np
import numpy.typing as npt
from gym import spaces


class SimpleUnitObservationWrapper_multiagent(gym.ObservationWrapper):
    """
    A simple state based observation to work with in pair with the SimpleUnitDiscreteController

    It contains info only on the first robot, the first factory you own, and some useful features. If there are no owned robots the observation is just zero.
    No information about the opponent is included. This will generate observations for all teams.

    Included features:
    - First robot's stats
    - distance vector to closest ice tile
    - distance vector to first factory

    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.observation_space = spaces.Box(-999, 999, shape=(13,))

    def observation(self, obs):
        return SimpleUnitObservationWrapper_multiagent.convert_obs(obs, self.env.state.env_cfg)

    # we make this method static so the submission/evaluation code can use this as well
    @staticmethod
    def convert_obs(obs: Dict[str, Any], env_cfg: Any) -> Dict[str, npt.NDArray]:
        def get_min4(locs, target, map_size):
            ret = [[-1, -1], [-1, -1], [-1, -1], [-1, -1]]
            min_dist2 = [1000000, 1000000, 1000000, 1000000]
            for tp in locs:
                cur_dist2 = (tp[0] - target[0]) ** 2 + (tp[1] - target[1]) ** 2
                if tp[0] < map_size // 2:
                    if tp[1] < map_size // 2:
                        if cur_dist2 < min_dist2[0]:
                            ret[0][0], ret[0][1] = tp[0]-target[0], tp[1]-target[1]
                            min_dist2[0] = cur_dist2
                    else:
                        if cur_dist2 < min_dist2[1]:
                            ret[1][0], ret[1][1] = tp[0]-target[0], tp[1]-target[1]
                            min_dist2[1] = cur_dist2
                else:
                    if tp[1] < map_size // 2:
                        if cur_dist2 < min_dist2[2]:
                            ret[2][0], ret[2][1] = tp[0]-target[0], tp[1]-target[1]
                            min_dist2[2] = cur_dist2
                    else:
                        if cur_dist2 < min_dist2[3]:
                            ret[3][0], ret[3][1] = tp[0]-target[0], tp[1]-target[1]
                            min_dist2[3] = cur_dist2
            return ret[0]+ret[1]+ret[2]+ret[3]
        shared_obs = obs["player_0"]
        space_info = np.ones([env_cfg.map_size+4,env_cfg.map_size+4])*10
        space_info[2:env_cfg.map_size+2,2:env_cfg.map_size+2] = shared_obs["board"]["rubble"]/200
        # make the return box
        ret = {}
        cargo_space = env_cfg.ROBOTS['HEAVY'].CARGO_SPACE
        battery_cap = env_cfg.ROBOTS['HEAVY'].BATTERY_CAPACITY
        # fill the self info
        for player_id, player_info in shared_obs['units'].items():
            for unit_id, unit_info in player_info.items():
                space_info[unit_info['pos'][0]+2, unit_info['pos'][1]+2] += 1
                ret[player_id] = {}
                ret[player_id][unit_id] = [-1 for _ in range(0,58)]
                ret[player_id][unit_id][0],ret[player_id][unit_id][1] = unit_info['pos'][0],unit_info['pos'][1]
                ret[player_id][unit_id][2], ret[player_id][unit_id][3], ret[player_id][unit_id][4] \
                    = unit_info['power']/battery_cap, \
                      unit_info['cargo']['ice']/cargo_space,\
                      unit_info['cargo']['ore']/cargo_space
        # fill the near space info
        ice_locs = np.argwhere(shared_obs["board"]["ice"] == 1)
        ore_locs = np.argwhere(shared_obs["board"]["ore"] == 1)
        for player_id, player_info in shared_obs['units'].items():
            for unit_id, unit_info in player_info.items():
                near_space = space_info[ret[player_id][unit_id][0]+2-2:ret[player_id][unit_id][0]+2+3,
                                        ret[player_id][unit_id][1]+2-2:ret[player_id][unit_id][1]+2+3]
                ret[player_id][unit_id][5:30] = near_space.reshape(-1)
                ret[player_id][unit_id][30:38] = get_min4(ice_locs, unit_info['pos'], env_cfg.map_size)
                ret[player_id][unit_id][38:46] = get_min4(ore_locs, unit_info['pos'], env_cfg.map_size)
                self_f_count = 0
                oppo_f_count = 0
                # add the factory info
                for p_id, p_info in shared_obs["factories"].items():
                    for f_id, f_info in p_info.items():
                        if p_id == player_id:
                            ret[player_id][unit_id][46+self_f_count:48+self_f_count] = f_info['pos'][0]-unit_info['pos'][0],f_info['pos'][1]-unit_info['pos'][1]
                            self_f_count += 2
                        else:
                            ret[player_id][unit_id][52 + oppo_f_count:54 + oppo_f_count] = f_info['pos'][0]-unit_info['pos'][0],f_info['pos'][1]-unit_info['pos'][1]
                            oppo_f_count += 2

        return ret
