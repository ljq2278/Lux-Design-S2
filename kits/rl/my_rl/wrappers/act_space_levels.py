import numpy as np


class ActSpaceUnit:
    move_act_dims = 4  # 0~3
    transfer_ice_act_dims = 1  # 4
    transfer_ore_act_dims = 1  # 5
    pickup_act_dims = 1  # 6
    dig_act_dims = 1  # 7
    no_op_dims = 1  # 8

    move_dim_high = move_act_dims
    transfer_ice_dim_high = move_dim_high + transfer_ice_act_dims
    transfer_ore_dim_high = transfer_ice_dim_high + transfer_ore_act_dims
    pickup_dim_high = transfer_ore_dim_high + pickup_act_dims
    dig_dim_high = pickup_dim_high + dig_act_dims
    no_op_dim_high = dig_dim_high + no_op_dims

    total_act_dims = no_op_dim_high

    def __init__(self, env_cfg):
        self.env_cfg = env_cfg

        return


class ActSpaceFactory:
    move_act_dims = 4  # 0~3
    transfer_ice_act_dims = 1  # 4
    transfer_ore_act_dims = 1  # 5
    pickup_act_dims = 1  # 6
    dig_act_dims = 1  # 7
    no_op_dims = 1  # 8

    move_dim_high = move_act_dims
    transfer_ice_dim_high = move_dim_high + transfer_ice_act_dims
    transfer_ore_dim_high = transfer_ice_dim_high + transfer_ore_act_dims
    pickup_dim_high = transfer_ore_dim_high + pickup_act_dims
    dig_dim_high = pickup_dim_high + dig_act_dims
    no_op_dim_high = dig_dim_high + no_op_dims

    total_act_dims = no_op_dim_high

    def __init__(self, env_cfg):
        self.env_cfg = env_cfg

        return
