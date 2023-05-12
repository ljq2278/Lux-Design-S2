import numpy as np


class ActSpaceUnit:
    move_dims = 4  # 0~3
    transfer_dims = 1  # 4
    pickup_dims = 1  # 5
    dig_dims = 1  # 6
    unit_noop_dims = 1  # 7

    move_high = move_dims
    transfer_high = move_high + transfer_dims
    pickup_high = transfer_high + pickup_dims
    dig_high = pickup_high + dig_dims
    unit_noop_high = dig_high + unit_noop_dims

    unit_dims = unit_noop_high

    def __init__(self, env_cfg):
        self.env_cfg = env_cfg

        return


class ActSpaceFactory:
    build_light_dims = 1  # 0
    build_heavy_dims = 1  # 1
    water_lichen_dims = 1  # 2
    factory_noop_dims = 1  # 3

    build_light_high = build_light_dims
    build_heavy_high = build_light_high + build_heavy_dims
    water_lichen_high = build_heavy_high + water_lichen_dims
    factory_noop_high = water_lichen_high + factory_noop_dims

    factory_dims = factory_noop_high

    def __init__(self, env_cfg):
        self.env_cfg = env_cfg

        return


class ActSpace(ActSpaceFactory, ActSpaceUnit):
    def __init__(self, env_cfg):
        ActSpaceFactory.__init__(self, env_cfg)
        ActSpaceUnit.__init__(self, env_cfg)
