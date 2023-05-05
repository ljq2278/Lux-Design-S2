import numpy as np


class ObsSpaceUnit:
    pos_dim_start = 0  # 0
    pos_dim = 2

    power_dim_start = pos_dim_start + pos_dim  # 2
    power_dim = 1
    ice_dim_start = power_dim_start + power_dim  # 3
    ice_dim = 1
    ore_dim_start = ice_dim_start + ice_dim  # 4
    ore_dim = 1

    near_space_start = ore_dim_start + ore_dim  # 5
    near_space = 5 * 5

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

        # self.mask = np.array([0 for _ in range(0, 2)] +
        #                      [1 for _ in range(0, 1)] +
        #                      [1 for _ in range(0, 2)] +
        #                      [0.5 if i != 12 else 1 for i in range(0, 25)] +
        #                      [1, 1, 1, 1, 1, 1, 1] +
        #                      [1 for _ in range(0, 2)], dtype=float)
        return

    @staticmethod
    def task_type_to_int(task_type):
        if task_type == 'ice':
            return 1
        elif task_type == 'ore':
            return 2
        elif task_type == 'rubble':
            return 3
        else:
            return -100

    @staticmethod
    def int_to_task_type(i):
        if i == 1:
            return 'ice'
        elif i == 2:
            return 'ore'
        elif i == 3:
            return 'rubble'
        else:
            return 'except'


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
    total_dims = power_dim_start + power_dim  # total 7

    def __init__(self, env_cfg):
        self.env_cfg = env_cfg
        self.normer = np.array([1 for _ in range(0, 7)])
        self.mask = np.array([1 for _ in range(0, 7)])
        return
