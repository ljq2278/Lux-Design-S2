import numpy as np


class ObsSpaceBoard:
    ice_dim_start = 0
    ice_dim = 1
    ore_dim_start = ice_dim_start + ice_dim  # 1
    ore_dim = 1
    rub_dim_start = ore_dim_start + ore_dim  # 2
    rub_dim = 1
    time_dim_start = rub_dim_start + rub_dim
    time_dim = 1
    left_step_dim_start = time_dim_start + time_dim
    left_step_dim = 1
    board_dim = left_step_dim_start + left_step_dim


class ObsSpaceUnit:
    power_dim_start = 0
    power_dim = 1
    ice_dim_start = power_dim_start + power_dim
    ice_dim = 1
    ore_dim_start = ice_dim_start + ice_dim  # 1
    ore_dim = 1
    typ_dim_start = ore_dim_start + ore_dim  # 1
    typ_dim = 1
    unit_dim = typ_dim_start + typ_dim
    # the 0,4,20,24,12 is is_in_target/is_at_home/target_dist/home_dist/target_home_water
    near_space_start = ore_dim_start + ore_dim  # 5
    near_space = 5 * 5
    is_in_target = near_space_start + 0
    is_at_home = near_space_start + 4
    target_dist = near_space_start + 20
    home_dist = near_space_start + 24
    # target_home_water = near_space_start + 12

    target_pos_start = near_space_start + near_space  # 30
    target_pos = 2

    target_factory_pos_start = target_pos_start + target_pos  # 32
    target_factory_pos = 2

    nearest_oppo_factory_pos_start = target_factory_pos_start + target_factory_pos  # 34
    nearest_oppo_factory_pos = 2

    day_or_night_start = nearest_oppo_factory_pos_start + nearest_oppo_factory_pos  # 36
    day_or_night = 1

    task_type_start = day_or_night_start + day_or_night  # 37    <=0: no task; 1: ice 2: ore 3: dig for plant
    task_type = 1

    transfered_start = task_type_start + task_type  # 38
    transfered = 1

    total_dims = transfered_start + transfered  # total 39 dim

    def __init__(self, env_cfg):
        self.env_cfg = env_cfg
        self.normer = np.array([self.env_cfg.map_size for _ in range(0, 2)] +
                               [self.env_cfg.ROBOTS['HEAVY'].BATTERY_CAPACITY for _ in range(0, 1)] +
                               [self.env_cfg.ROBOTS['HEAVY'].CARGO_SPACE / 100 for _ in range(0, 2)] +
                               [self.env_cfg.MAX_RUBBLE for _ in range(0, 25)] +
                               [self.env_cfg.map_size / 10 for _ in range(0, 6)] +
                               [1 for _ in range(0, 2)] +
                               [50 for _ in range(0, 1)], dtype=float)
        self.normer[ObsSpaceUnit.is_in_target] = 1
        self.normer[ObsSpaceUnit.is_at_home] = 1
        self.normer[ObsSpaceUnit.target_dist] = 1
        self.normer[ObsSpaceUnit.home_dist] = 1

        return


class ObsSpaceFactory:
    pos_dim_start = 0  # 0
    pos_dim = 2
    water_dim_start = pos_dim_start + pos_dim  # 2
    water_dim = 1
    metal_dim_start = water_dim_start + water_dim  # 3
    metal_dim = 1
    ice_dim_start = metal_dim_start + metal_dim  # 4
    ice_dim = 1
    ore_dim_start = ice_dim_start + ice_dim  # 5
    ore_dim = 1
    power_dim_start = ore_dim_start + ore_dim  # 6
    power_dim = 1
    lichen_dim_start = power_dim_start + power_dim  # 7
    lichen_dim = 1
    left_step_dim_start = lichen_dim_start + lichen_dim  # 8
    left_step_dim = 1

    heavy_build_dim_start = left_step_dim_start + left_step_dim  # 9
    heavy_build_dim = 1

    plant_pos_dim_start = heavy_build_dim_start + heavy_build_dim  # 10
    plant_pos_dim = 1

    ice_prob_dim_start = plant_pos_dim_start + plant_pos_dim  # 11
    ice_prob_dim = 1

    ore_prob_dim_start = ice_prob_dim_start + ice_prob_dim  # 12
    ore_prob_dim = 1

    rub_prob_dim_start = ore_prob_dim_start + ore_prob_dim  # 13
    rub_prob_dim = 1

    nearest_rubble_dist_start = rub_prob_dim_start + rub_prob_dim  # 14
    nearest_rubble_dist = 1

    nearest_rubble_value_start = nearest_rubble_dist_start + nearest_rubble_dist  # 15
    nearest_rubble_value = 1

    total_dims = nearest_rubble_value_start + nearest_rubble_value  # total 16

    def __init__(self, env_cfg):
        self.env_cfg = env_cfg
        self.normer = np.array(
            [10 for _ in range(0, 2)] +
            [100 for _ in range(2, 7)] +
            [10 for _ in range(7, 11)] +
            [1 for _ in range(11, 14)] +
            [10 for _ in range(14, self.total_dims)]
        )
        return
