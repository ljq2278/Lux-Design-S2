from typing import Any, Dict

import gym
import numpy as np
import numpy.typing as npt
from gym import spaces

class SimpleUnitObservationWrapper(gym.ObservationWrapper):
    """
    A simple state based observation to work with in pair with the SimpleUnitDiscreteController

    It contains info only on the first robot, the first factory you own, and some useful features. If there are no owned robots the observation is just zero.
    No information about the opponent is included. This will generate observations for all teams.

    Included features:
    - First robot's stats
    - distance vector to closest ice tile
    - distance vector to first factory

    """
    HEIGHT, WIDTH = 48, 48
    N_CHANNELS = 6
    cargo_and_power_ind = {
        'ice': 0, 'ore': 1, 'water': 2, 'metal': 3, 'power': 4
    }

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        # self.observation_space = spaces.Box(-999, 999, shape=(13,))
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.N_CHANNELS, self.HEIGHT, self.WIDTH),
                                            dtype=np.uint8)

    def observation(self, obs):
        return SimpleUnitObservationWrapper.convert_obs(obs, self.env.state.env_cfg)

    # we make this method static so the submission/evaluation code can use this as well
    @staticmethod
    def convert_obs(obs: Dict[str, Any], env_cfg: Any) -> Dict[str, npt.NDArray]:
        shared_obs = obs["player_0"]
        maps = [
            shared_obs['board']['rubble'],  # 0
            shared_obs['board']['ore'],  # 1
            shared_obs['board']['ice'],  # 2
            shared_obs['board']['lichen'],  # 3
            shared_obs['board']['lichen_strains'],  # 4
            shared_obs['board']['valid_spawns_mask'],  # 5
            np.zeros(shape=[SimpleUnitObservationWrapper.HEIGHT, SimpleUnitObservationWrapper.WIDTH]),
            # player_0 light ice#6
            np.zeros(shape=[SimpleUnitObservationWrapper.HEIGHT, SimpleUnitObservationWrapper.WIDTH]),
            # player_0 light ore#7
            np.zeros(shape=[SimpleUnitObservationWrapper.HEIGHT, SimpleUnitObservationWrapper.WIDTH]),
            # player_0 light water#8
            np.zeros(shape=[SimpleUnitObservationWrapper.HEIGHT, SimpleUnitObservationWrapper.WIDTH]),
            # player_0 light metal#9
            np.zeros(shape=[SimpleUnitObservationWrapper.HEIGHT, SimpleUnitObservationWrapper.WIDTH]),
            # player_0 light power#10
            np.zeros(shape=[SimpleUnitObservationWrapper.HEIGHT, SimpleUnitObservationWrapper.WIDTH]),
            # player_1 light ice#11
            np.zeros(shape=[SimpleUnitObservationWrapper.HEIGHT, SimpleUnitObservationWrapper.WIDTH]),
            # player_1 light ore#12
            np.zeros(shape=[SimpleUnitObservationWrapper.HEIGHT, SimpleUnitObservationWrapper.WIDTH]),
            # player_1 light water#13
            np.zeros(shape=[SimpleUnitObservationWrapper.HEIGHT, SimpleUnitObservationWrapper.WIDTH]),
            # player_1 light metal#14
            np.zeros(shape=[SimpleUnitObservationWrapper.HEIGHT, SimpleUnitObservationWrapper.WIDTH]),
            # player_1 light power#15
            np.zeros(shape=[SimpleUnitObservationWrapper.HEIGHT, SimpleUnitObservationWrapper.WIDTH]),
            # player_0 heavy ice#16
            np.zeros(shape=[SimpleUnitObservationWrapper.HEIGHT, SimpleUnitObservationWrapper.WIDTH]),
            # player_0 heavy ore#17
            np.zeros(shape=[SimpleUnitObservationWrapper.HEIGHT, SimpleUnitObservationWrapper.WIDTH]),
            # player_0 heavy water#18
            np.zeros(shape=[SimpleUnitObservationWrapper.HEIGHT, SimpleUnitObservationWrapper.WIDTH]),
            # player_0 heavy metal#19
            np.zeros(shape=[SimpleUnitObservationWrapper.HEIGHT, SimpleUnitObservationWrapper.WIDTH]),
            # player_0 heavy power#20
            np.zeros(shape=[SimpleUnitObservationWrapper.HEIGHT, SimpleUnitObservationWrapper.WIDTH]),
            # player_1 heavy ice#21
            np.zeros(shape=[SimpleUnitObservationWrapper.HEIGHT, SimpleUnitObservationWrapper.WIDTH]),
            # player_1 heavy ore#22
            np.zeros(shape=[SimpleUnitObservationWrapper.HEIGHT, SimpleUnitObservationWrapper.WIDTH]),
            # player_1 heavy water#23
            np.zeros(shape=[SimpleUnitObservationWrapper.HEIGHT, SimpleUnitObservationWrapper.WIDTH]),
            # player_1 heavy metal#24
            np.zeros(shape=[SimpleUnitObservationWrapper.HEIGHT, SimpleUnitObservationWrapper.WIDTH]),
            # player_1 heavy power#25
        ]
        units = shared_obs['units']
        for player, player_units in units.items():
            dim_stride = 0 if player == 'player_0' else 1
            for unit_name, info in player_units.items():
                if info['unit_type'] == 'LIGHT':
                    for k, v in info['cargo'].items():
                        maps[6 + dim_stride * 5 + SimpleUnitObservationWrapper.cargo_and_power_ind[k]][
                            info['pos'][0], info['pos'][1]] = v
                    maps[6 + dim_stride * 5 + SimpleUnitObservationWrapper.cargo_and_power_ind['power']][
                        info['pos'][0], info['pos'][1]] = info['power']
                else:
                    for k, v in info['cargo'].items():
                        maps[16 + dim_stride * 5 + SimpleUnitObservationWrapper.cargo_and_power_ind[k]][
                            info['pos'][0], info['pos'][1]] = v
                    maps[16 + dim_stride * 5 + SimpleUnitObservationWrapper.cargo_and_power_ind['power']][
                        info['pos'][0], info['pos'][1]] = info['power']
        # if observer is player_1, for use the same net ,we should change the dim order
        maps_player1 = maps[:6]+maps[11:16]+maps[6:11]+maps[21:26]+maps[16:21]
        maps = np.array(maps[:SimpleUnitObservationWrapper.N_CHANNELS])
        maps_player1 = np.array(maps_player1[:SimpleUnitObservationWrapper.N_CHANNELS])
        # return maps
        return {"player_0": maps,"player_1": maps_player1}
