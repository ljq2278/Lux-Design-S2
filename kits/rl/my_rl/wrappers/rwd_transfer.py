from typing import Any, Dict

import gym
import numpy as np
import numpy.typing as npt
from gym import spaces
# import matplotlib.pyplot as plt
from lux.config import EnvConfig
from wrappers.obs_space_levels import ObsSpaceUnit, ObsSpaceFactory
from wrappers.act_space_levels import ActSpaceUnit, ActSpaceFactory, ActSpaceFactoryDemand


class RwdTransfer:

    def __init__(self, env, env_cfg: EnvConfig, debug=False, density=True) -> None:
        self.env_cfg = env_cfg
        self.env = env
        self.debug = debug
        self.density = density
        self.reward_collect = {
            'on the way target': 0,
            'leave the way target': 0,
            'dig out rubble on target': 0,
            'dig target': 0,
            'on the way rubble': 0,
            'leave the way rubble': 0,
            'on the way home with target': 0,
            'leave the way home with target': 0,
            'collision happen': 0,
            'transfer target': 0,
            'transfer ice': 0,
            'transfer ore': 0,
            'dig out target rubble': 0,
            'low power charged': 0,
        }
        return

    def raw_to_wrap(self, ori_reward, done):
        if done:
            return ori_reward
        else:
            return 10

