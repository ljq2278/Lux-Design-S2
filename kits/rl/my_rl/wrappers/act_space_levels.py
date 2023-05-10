import numpy as np


class ActSpaceUnit:
    move_act_dims = 4  # 0~3
    transfer_target_dims = 1  # 4
    pickup_act_dims = 1  # 5
    dig_act_dims = 1  # 6
    no_op_dims = 1  # 7

    move_dim_high = move_act_dims
    transfer_target_dim_high = move_dim_high + transfer_target_dims
    pickup_dim_high = transfer_target_dim_high + pickup_act_dims
    dig_dim_high = pickup_dim_high + dig_act_dims
    no_op_dim_high = dig_dim_high + no_op_dims

    total_act_dims = no_op_dim_high

    def __init__(self, env_cfg):
        self.env_cfg = env_cfg

        return


class ActSpaceFactory:
    build_light_dims = 1  # 0
    build_heavy_dims = 1  # 1
    water_lichen_dims = 1  # 2
    no_op_dims = 1  # 3

    build_light_high = build_light_dims
    build_heavy_high = build_light_high + build_heavy_dims
    water_lichen_high = build_heavy_high + water_lichen_dims
    no_op_dim_high = water_lichen_high + no_op_dims

    total_act_dims = no_op_dim_high

    def __init__(self, env_cfg):
        self.env_cfg = env_cfg

        return


class ActSpaceFactoryDemand:
    incre_ice_decre_ore = 1  # 0
    incre_ice_decre_rub = 1  # 1
    incre_ore_decre_ice = 1  # 0
    incre_ore_decre_rub = 1  # 1
    incre_rub_decre_ice = 1  # 0
    incre_rub_decre_ore = 1  # 1
    keep_the_same = 1

    incre_ice_decre_ore_high = 0 + incre_ice_decre_ore
    incre_ice_decre_rub_high = incre_ice_decre_ore_high + incre_ice_decre_rub
    incre_ore_decre_ice_high = incre_ice_decre_rub_high + incre_ore_decre_ice
    incre_ore_decre_rub_high = incre_ore_decre_ice_high + incre_ore_decre_rub
    incre_rub_decre_ice_high = incre_ore_decre_rub_high + incre_rub_decre_ice
    incre_rub_decre_ore_high = incre_rub_decre_ice_high + incre_rub_decre_ore
    keep_the_same_high = incre_rub_decre_ore_high + keep_the_same

    total_act_dims = keep_the_same_high

    @staticmethod
    def demand_id_to_increment(ind):
        if ind < ActSpaceFactoryDemand.incre_ice_decre_ore_high:
            return {'ice': 0.1, 'ore': -0.1, 'rubble': 0}
        elif ind < ActSpaceFactoryDemand.incre_ice_decre_rub_high:
            return {'ice': 0.1, 'ore': 0, 'rubble': -0.1}
        elif ind < ActSpaceFactoryDemand.incre_ore_decre_ice_high:
            return {'ice': -0.1, 'ore': 0.1, 'rubble': 0}
        elif ind < ActSpaceFactoryDemand.incre_ore_decre_rub_high:
            return {'ice': 0, 'ore': 0.1, 'rubble': -0.1}
        elif ind < ActSpaceFactoryDemand.incre_rub_decre_ice_high:
            return {'ice': -0.1, 'ore': 0, 'rubble': 0.1}
        elif ind < ActSpaceFactoryDemand.incre_rub_decre_ore_high:
            return {'ice': 0, 'ore': -0.1, 'rubble': 0.1}
        else:
            return {'ice': 0, 'ore': 0, 'rubble': 0}

    def __init__(self, env_cfg):
        self.env_cfg = env_cfg

        return
