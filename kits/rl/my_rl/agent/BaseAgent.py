"""
This file is where your agent's logic is kept. Define a bidding policy, factory placement policy, as well as a policy for playing the normal phase of the game

The tutorial will learn an RL agent to play the normal phase and use heuristics for the other two phases.

Note that like the other kits, you can only debug print to standard error e.g. print("message", file=sys.stderr)
"""

import os.path as osp
import sys
import numpy as np
import torch as th
# from stable_baselines3.ppo import PPO
from lux.config import EnvConfig
from lux.kit import obs_to_game_state, GameState, EnvConfig
from lux.utils import direction_to, my_turn_to_place_factory
from copy import deepcopy
from typing import List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.optim import Adam

# change this to use weights stored elsewhere
# make sure the model weights are submitted with the other code files
# any files in the logs folder are not necessary. Make sure to exclude the .zip extension here
MODEL_WEIGHTS_RELATIVE_PATH = "./best_model"



class Agent:
    def __init__(self, player: str, env_cfg: EnvConfig) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        np.random.seed(0)
        self.env_cfg: EnvConfig = env_cfg


class EarlyRuleAgent(Agent):
    def __init__(self, player, env_cfg):
        super().__init__(player, env_cfg)

        self.faction_names = {
            'player_0': 'AlphaStrike',
            'player_1': 'MotherMars'
        }

    def early_setup(self, step: int, obs, remainingOverageTime: int = 60):
        '''
        Early Phase
        '''

        actions = dict()
        if step == 0:
            # Declare faction
            actions['faction'] = self.faction_names[self.player]
            actions['bid'] = 0  # Learnable
        else:
            # Factory placement period
            # optionally convert observations to python objects with utility functions
            game_state = obs_to_game_state(step, self.env_cfg, obs)
            opp_factories = [f.pos for _, f in game_state.factories[self.opp_player].items()]
            my_factories = [f.pos for _, f in game_state.factories[self.player].items()]

            # how much water and metal you have in your starting pool to give to new factories
            water_left = game_state.teams[self.player].water
            metal_left = game_state.teams[self.player].metal

            # how many factories you have left to place
            factories_to_place = game_state.teams[self.player].factories_to_place
            my_turn_to_place = my_turn_to_place_factory(game_state.teams[self.player].place_first, step)
            if factories_to_place > 0 and my_turn_to_place:
                # we will spawn our factory in a random location with 100 metal n water (learnable)
                potential_spawns = np.array(list(zip(*np.where(obs["board"]["valid_spawns_mask"] == 1))))

                ice_map = game_state.board.ice
                ore_map = game_state.board.ore
                ice_tile_locations = np.argwhere(ice_map == 1)  # numpy position of every ice tile
                ore_tile_locations = np.argwhere(ore_map == 1)  # numpy position of every ice tile

                min_dist = 10e6
                best_loc = potential_spawns[0]

                d_rubble = 10

                for loc in potential_spawns:

                    ice_tile_distances = np.mean((ice_tile_locations - loc) ** 2, 1)
                    ore_tile_distances = np.mean((ore_tile_locations - loc) ** 2, 1)
                    density_rubble = np.mean(
                        obs["board"]["rubble"][max(loc[0] - d_rubble, 0):min(loc[0] + d_rubble, 47),
                        max(loc[1] - d_rubble, 0):max(loc[1] + d_rubble, 47)])

                    closes_opp_factory_dist = 0
                    if len(opp_factories) >= 1:
                        closes_opp_factory_dist = np.min(np.mean((np.array(opp_factories) - loc) ** 2, 1))
                    closes_my_factory_dist = 0
                    if len(my_factories) >= 1:
                        closes_my_factory_dist = np.min(np.mean((np.array(my_factories) - loc) ** 2, 1))

                    minimum_ice_dist = np.min(ice_tile_distances) * 10 + 0.01 * np.min(
                        ore_tile_distances) + 10 * density_rubble / (
                                           d_rubble) - closes_opp_factory_dist * 0.1 + closes_opp_factory_dist * 0.01

                    if minimum_ice_dist < min_dist:
                        min_dist = minimum_ice_dist
                        best_loc = loc

                #                 spawn_loc = potential_spawns[np.random.randint(0, len(potential_spawns))]
                spawn_loc = best_loc
                actions['spawn'] = spawn_loc
                #                 actions['metal']=metal_left
                #                 actions['water']=water_left
                actions['metal'] = min(self.env_cfg.INIT_WATER_METAL_PER_FACTORY, metal_left)
                actions['water'] = min(self.env_cfg.INIT_WATER_METAL_PER_FACTORY, metal_left)

        return actions


