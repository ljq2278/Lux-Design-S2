import numpy as np


class ObsSpaceBoard:
    def __init__(self, start_dim=0):
        self.b_ice_dim_start = start_dim
        self.b_ice_dim = 1
        self.b_ore_dim_start = self.b_ice_dim_start + self.b_ice_dim  # 1
        self.b_ore_dim = 1
        self.b_rub_dim_start = self.b_ore_dim_start + self.b_ore_dim  # 2
        self.b_rub_dim = 1
        self.b_time_dim_start = self.b_rub_dim_start + self.b_rub_dim
        self.b_time_dim = 1
        self.b_left_step_dim_start = self.b_time_dim_start + self.b_time_dim
        self.b_left_step_dim = 1
        self.b_dims = self.b_left_step_dim_start + self.b_left_step_dim - start_dim
        self.b_normer = [1, 1, 100, 50, 50]


class ObsSpaceFactory:
    def __init__(self, start_dim=0):
        self.f_pos_dim_start = start_dim
        self.f_pos_dim = 1
        self.f_water_dim_start = self.f_pos_dim_start + self.f_pos_dim
        self.f_water_dim = 1
        self.f_metal_dim_start = self.f_water_dim_start + self.f_water_dim
        self.f_metal_dim = 1
        self.f_ice_dim_start = self.f_metal_dim_start + self.f_metal_dim
        self.f_ice_dim = 1
        self.f_ore_dim_start = self.f_ice_dim_start + self.f_ice_dim
        self.f_ore_dim = 1
        self.f_power_dim_start = self.f_ore_dim_start + self.f_ore_dim
        self.f_power_dim = 1
        self.f_grow_lichen_dim_start = self.f_power_dim_start + self.f_power_dim
        self.f_grow_lichen_dim = 1
        self.f_connected_lichen_dim_start = self.f_grow_lichen_dim_start + self.f_grow_lichen_dim
        self.f_connected_lichen_dim = 1
        self.f_dims = self.f_connected_lichen_dim_start + self.f_connected_lichen_dim - start_dim
        self.f_normer = [1, 10, 10, 100, 100, 1000, 1, 1]


class ObsSpaceOppoFactory:
    def __init__(self, start_dim=0):
        self.of_pos_dim_start = start_dim
        self.of_pos_dim = 1
        self.of_water_dim_start = self.of_pos_dim_start + self.of_pos_dim
        self.of_water_dim = 1
        self.of_metal_dim_start = self.of_water_dim_start + self.of_water_dim
        self.of_metal_dim = 1
        self.of_ice_dim_start = self.of_metal_dim_start + self.of_metal_dim
        self.of_ice_dim = 1
        self.of_ore_dim_start = self.of_ice_dim_start + self.of_ice_dim
        self.of_ore_dim = 1
        self.of_power_dim_start = self.of_ore_dim_start + self.of_ore_dim
        self.of_power_dim = 1
        self.of_grow_lichen_dim_start = self.of_power_dim_start + self.of_power_dim
        self.of_grow_lichen_dim = 1
        self.of_connected_lichen_dim_start = self.of_grow_lichen_dim_start + self.of_grow_lichen_dim
        self.of_connected_lichen_dim = 1
        self.of_dims = self.of_connected_lichen_dim_start + self.of_connected_lichen_dim - start_dim
        self.of_normer = [1, 10, 10, 100, 100, 1000, 1, 1]


class ObsSpaceUnit:
    def __init__(self, start_dim=0):
        self.u_pos_dim_start = start_dim
        self.u_pos_dim = 1
        self.u_power_dim_start = self.u_pos_dim_start + self.u_pos_dim
        self.u_power_dim = 1
        self.u_ice_dim_start = self.u_power_dim_start + self.u_power_dim
        self.u_ice_dim = 1
        self.u_ore_dim_start = self.u_ice_dim_start + self.u_ice_dim
        self.u_ore_dim = 1
        self.u_typ_dim_start = self.u_ore_dim_start + self.u_ore_dim
        self.u_typ_dim = 1
        self.u_dims = self.u_typ_dim_start + self.u_typ_dim - start_dim
        self.u_normer = [1, 100, 10, 10, 1]


class ObsSpaceOppoUnit:
    def __init__(self, start_dim=0):
        self.ou_pos_dim_start = start_dim
        self.ou_pos_dim = 1
        self.ou_power_dim_start = self.ou_pos_dim_start + self.ou_pos_dim
        self.ou_power_dim = 1
        self.ou_ice_dim_start = self.ou_power_dim_start + self.ou_power_dim
        self.ou_ice_dim = 1
        self.ou_ore_dim_start = self.ou_ice_dim_start + self.ou_ice_dim
        self.ou_ore_dim = 1
        self.ou_typ_dim_start = self.ou_ore_dim_start + self.ou_ore_dim
        self.ou_typ_dim = 1
        self.ou_dims = self.ou_typ_dim_start + self.ou_typ_dim - start_dim
        self.ou_normer = [1, 100, 10, 10, 1]


class ObsSpace(ObsSpaceBoard, ObsSpaceUnit, ObsSpaceFactory, ObsSpaceOppoUnit, ObsSpaceOppoFactory):

    def __init__(self, env_cfg):
        self.env_cfg = env_cfg
        ObsSpaceBoard.__init__(self, start_dim=0)
        ObsSpaceFactory.__init__(self, start_dim=self.b_dims)
        ObsSpaceOppoFactory.__init__(self, start_dim=self.b_dims + self.f_dims)
        ObsSpaceUnit.__init__(self, start_dim=self.b_dims + self.f_dims + self.of_dims)
        ObsSpaceOppoUnit.__init__(self, start_dim=self.b_dims + self.f_dims + self.of_dims + self.u_dims)
        self.total_dims = self.b_dims + self.f_dims + self.of_dims + self.u_dims + self.ou_dims
        self.normer = np.array(self.b_normer + self.f_normer + self.of_normer + self.u_normer + self.ou_normer, dtype=float)


class ObsSpaceStat:
    def __init__(self, start_dim=0):
        self.s_rub_dim_start = start_dim
        self.s_rub_dim = 1
        self.s_ice_dim_start = self.s_rub_dim_start + self.s_rub_dim
        self.s_ice_dim = 1
        self.s_ore_dim_start = self.s_ice_dim_start + self.s_ice_dim
        self.s_ore_dim = 1
        self.s_water_dim_start = self.s_ore_dim_start + self.s_ore_dim
        self.s_water_dim = 1
        self.s_metal_dim_start = self.s_water_dim_start + self.s_water_dim
        self.s_metal_dim = 1
        self.s_dims = self.s_metal_dim_start + self.s_metal_dim - start_dim
        self.s_normer = [1, 1, 1, 1, 1]