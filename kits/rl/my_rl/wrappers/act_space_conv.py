import numpy as np


class ActSpaceFactory:

    def __init__(self, start_dim=0):
        self.f_build_light_dim_start = start_dim
        self.f_build_light_dim = 1
        self.f_build_heavy_dim_start = self.f_build_light_dim_start + self.f_build_light_dim
        self.f_build_heavy_dim = 1
        self.f_water_lichen_dim_start = self.f_build_heavy_dim_start + self.f_build_heavy_dim
        self.f_water_lichen_dim = 1
        self.f_noop_dim_start = self.f_water_lichen_dim_start + self.f_water_lichen_dim
        self.f_noop_dim = 1
        self.f_dims = self.f_noop_dim_start + self.f_noop_dim - start_dim


class ActSpaceUnit:

    def __init__(self, start_dim=0):
        self.u_move_dim_start = start_dim
        self.u_move_dim = 4
        self.u_pickup_dim_start = self.u_move_dim_start + self.u_move_dim
        self.u_pickup_dim = 4  # 100p, 300p, 1000p, 2000p
        self.u_transfer_dim_start = self.u_pickup_dim_start + self.u_pickup_dim
        self.u_transfer_dim = 2 * 5  # ice 5 direct/ ore 5 direct
        self.u_dig_dim_start = self.u_transfer_dim_start + self.u_transfer_dim
        self.u_dig_dim = 1
        self.u_noop_dim_start = self.u_dig_dim_start + self.u_dig_dim
        self.u_noop_dim = 1
        self.u_dims = self.u_noop_dim_start + self.u_noop_dim - start_dim

# class ActSpace(ActSpaceFactory, ActSpaceUnit):
#
#     def __init__(self, env_cfg):
#         self.env_cfg = env_cfg
#         ActSpaceFactory.__init__(self, start_dim=0)
#         ActSpaceUnit.__init__(self, start_dim=self.f_dims)
#         self.total_dims = self.f_dims + self.u_dims
