from typing import Any, Dict
import copy
import gym
import numpy as np
import numpy.typing as npt
from gym import spaces
from lux.config import EnvConfig
from wrappers.obs_space_conv import ObsSpace
from wrappers.act_space_levels import ActSpaceUnit, ActSpaceFactory
from ppo.FactoryAgent import Factory_Agent


class ObsTransfer:

    def __init__(self, env, env_cfg: EnvConfig) -> None:
        self.env = env
        self.env_cfg = env_cfg
        self.obs_space = ObsSpace(env_cfg)
        self.oppo_dict = {
            'player_0': 'player_1',
            'player_1': 'player_0'
        }
        self.typ_dict = {
            'LIGHT': 1,
            'HEAVY': 2
        }

    def raw_to_wrap(self, raw_obs: Dict[str, Any], env_state, left_step):
        ret = {}
        ########################################################## init and config board ###########################################
        for p_id in ['player_0', 'player_1']:
            matrix_features = np.zeros([self.obs_space.total_dims, self.env_cfg.map_size, self.env_cfg.map_size], dtype=float)  # CxHxW
            ret[p_id] = matrix_features
            ret[p_id][self.obs_space.b_ice_dim_start][:, :] = env_state.board.ice
            ret[p_id][self.obs_space.b_ore_dim_start][:, :] = env_state.board.ore
            ret[p_id][self.obs_space.b_rub_dim_start][:, :] = env_state.board.rubble
            ret[p_id][self.obs_space.b_time_dim_start][:, :] = raw_obs['real_env_steps'] % self.env_cfg.CYCLE_LENGTH
            ret[p_id][self.obs_space.b_left_step_dim_start][:, :] = left_step
        ########################################################## set factories ###########################################
        for p_id, p_finfos in raw_obs['factories'].items():
            for f_id, f_info in p_finfos.items():
                ###################################################### set player f #########################################
                ret[p_id][self.obs_space.f_pos_dim_start][f_info['pos'][0], f_info['pos'][1]] = 1
                ret[p_id][self.obs_space.f_ice_dim_start][f_info['pos'][0], f_info['pos'][1]] = f_info['cargo']['ice']
                ret[p_id][self.obs_space.f_ore_dim_start][f_info['pos'][0], f_info['pos'][1]] = f_info['cargo']['ore']
                ret[p_id][self.obs_space.f_water_dim_start][f_info['pos'][0], f_info['pos'][1]] = f_info['cargo']['water']
                ret[p_id][self.obs_space.f_metal_dim_start][f_info['pos'][0], f_info['pos'][1]] = f_info['cargo']['metal']
                ret[p_id][self.obs_space.f_power_dim_start][f_info['pos'][0], f_info['pos'][1]] = f_info['power']
                ###################################################### set oppo player f #########################################
                ret[self.oppo_dict[p_id]][self.obs_space.of_pos_dim_start][f_info['pos'][0], f_info['pos'][1]] = 1
                ret[self.oppo_dict[p_id]][self.obs_space.of_ice_dim_start][f_info['pos'][0], f_info['pos'][1]] = f_info['cargo']['ice']
                ret[self.oppo_dict[p_id]][self.obs_space.of_ore_dim_start][f_info['pos'][0], f_info['pos'][1]] = f_info['cargo']['ore']
                ret[self.oppo_dict[p_id]][self.obs_space.of_water_dim_start][f_info['pos'][0], f_info['pos'][1]] = f_info['cargo']['water']
                ret[self.oppo_dict[p_id]][self.obs_space.of_metal_dim_start][f_info['pos'][0], f_info['pos'][1]] = f_info['cargo']['metal']
                ret[self.oppo_dict[p_id]][self.obs_space.of_power_dim_start][f_info['pos'][0], f_info['pos'][1]] = f_info['power']
                ############################################################# lichen setting ###################################
                for pos in env_state.factories[p_id][f_id].grow_lichen_positions:
                    ret[p_id][self.obs_space.f_grow_lichen_dim_start][pos[0], pos[1]] = env_state.board.lichen_strains[pos[0], pos[1]] + 1
                    ret[self.oppo_dict[p_id]][self.obs_space.f_grow_lichen_dim_start][pos[0], pos[1]] = env_state.board.lichen_strains[pos[0], pos[1]] + 1
                for pos in env_state.factories[p_id][f_id].connected_lichen_positions:
                    ret[p_id][self.obs_space.f_connected_lichen_dim_start][pos[0], pos[1]] = env_state.board.lichen[pos[0], pos[1]] + 1
                    ret[self.oppo_dict[p_id]][self.obs_space.f_connected_lichen_dim_start][pos[0], pos[1]] = env_state.board.lichen[pos[0], pos[1]] + 1
        ###################################################### set unit #########################################
        for p_id, p_uinfos in raw_obs['units'].items():
            for u_id, u_info in p_uinfos.items():
                ###################################################### set player u #########################################
                ret[p_id][self.obs_space.u_pos_dim_start][u_info['pos'][0], u_info['pos'][1]] = 1
                ret[p_id][self.obs_space.u_ice_dim_start][u_info['pos'][0], u_info['pos'][1]] = u_info['cargo']['ice']
                ret[p_id][self.obs_space.u_ore_dim_start][u_info['pos'][0], u_info['pos'][1]] = u_info['cargo']['ore']
                ret[p_id][self.obs_space.u_power_dim_start][u_info['pos'][0], u_info['pos'][1]] = u_info['power']
                ret[p_id][self.obs_space.u_typ_dim_start][u_info['pos'][0], u_info['pos'][1]] = self.typ_dict[u_info['unit_type']]
                ###################################################### set oppo player u #########################################
                ret[self.oppo_dict[p_id]][self.obs_space.u_pos_dim_start][u_info['pos'][0], u_info['pos'][1]] = 1
                ret[self.oppo_dict[p_id]][self.obs_space.u_ice_dim_start][u_info['pos'][0], u_info['pos'][1]] = u_info['cargo']['ice']
                ret[self.oppo_dict[p_id]][self.obs_space.u_ore_dim_start][u_info['pos'][0], u_info['pos'][1]] = u_info['cargo']['ore']
                ret[self.oppo_dict[p_id]][self.obs_space.u_power_dim_start][u_info['pos'][0], u_info['pos'][1]] = u_info['power']
                ret[self.oppo_dict[p_id]][self.obs_space.u_typ_dim_start][u_info['pos'][0], u_info['pos'][1]] = self.typ_dict[u_info['unit_type']]
        return ret
