import numpy as np


class ObsSpaceUnit:
    pos_dim_start = 0  # 0
    pos_dim = 2
    power_dim_start = pos_dim_start + pos_dim  # 2
    power_dim = 1
    cargo_dim_start = power_dim_start + power_dim  # 3
    cargo_dim = 2
    near_space_start = cargo_dim_start + cargo_dim  # 5
    near_space = 5 * 5

    target_pos_start = near_space_start + near_space  # 30
    target_pos = 2

    target_factory_pos_start = target_pos_start + target_pos  # 32
    target_factory_pos = 2

    nearest_oppo_factory_pos_start = target_factory_pos_start + target_factory_pos  # 34
    nearest_oppo_factory_pos = 2

    day_or_night_start = nearest_oppo_factory_pos_start + nearest_oppo_factory_pos  # 36
    day_or_night = 1

    total_dims = day_or_night_start + day_or_night  # total 37 dim

    def __init__(self, env_cfg):
        self.env_cfg = env_cfg
        self.normer = np.array([self.env_cfg.map_size for _ in range(0, 2)] +
                               [self.env_cfg.ROBOTS['HEAVY'].BATTERY_CAPACITY for _ in range(0, 1)] +
                               [self.env_cfg.ROBOTS['HEAVY'].CARGO_SPACE / 100 for _ in range(0, 2)] +
                               [self.env_cfg.MAX_RUBBLE for _ in range(0, 25)] +
                               [self.env_cfg.map_size / 10 for _ in range(0, 6)] +
                               [1 for _ in range(0, 1)])

        self.mask = np.array([0 for _ in range(0, 2)] +
                             [1 for _ in range(0, 1)] +
                             [1 for _ in range(0, 2)] +
                             [0.5 if i != 12 else 1 for i in range(0, 25)] +
                             [1, 1, 1, 1, 1, 1] +
                             [1 for _ in range(0, 1)])
        return


class ObsSpaceFactory:
    def __init__(self):
        pass
