import numpy as np


class ObsSpace:
    pos_dim_start = 0
    pos_dim = 2
    power_dim_start = pos_dim_start + pos_dim  # 2
    power_dim = 1
    cargo_dim_start = power_dim_start + power_dim  # 3
    cargo_dim = 2
    near_space_start = cargo_dim_start + cargo_dim  # 5
    near_space = 5 * 5

    nearest_ice_pos_start = near_space_start + near_space  # 30
    nearest_ice_pos = 2
    nearest_ore_pos_start = nearest_ice_pos_start + nearest_ice_pos  # 32
    nearest_ore_pos = 2

    nearest_factory_pos_start = nearest_ore_pos_start + nearest_ore_pos  # 34
    nearest_factory_pos = 2

    nearest_oppo_factory_pos_start = nearest_factory_pos_start + nearest_factory_pos  # 36
    nearest_oppo_factory_pos = 2

    nearest_factory_water_start = nearest_oppo_factory_pos_start + nearest_oppo_factory_pos  # 38
    nearest_factory_water = 1

    day_or_night_start = nearest_factory_water_start + nearest_factory_water  # 39
    day_or_night = 1

    has_ice_start = day_or_night_start + day_or_night  # 40
    has_ice = 1

    has_ore_start = has_ice_start + has_ice  # 41
    has_ore = 1

    total_dims = has_ore_start + has_ore  # 42

    def __init__(self, env_cfg):
        self.env_cfg = env_cfg
        self.normer = np.array([self.env_cfg.map_size for _ in range(0, 2)] +
                               [self.env_cfg.ROBOTS['HEAVY'].BATTERY_CAPACITY for _ in range(0, 1)] +
                               [self.env_cfg.ROBOTS['HEAVY'].CARGO_SPACE for _ in range(0, 2)] +
                               [self.env_cfg.MAX_RUBBLE for _ in range(0, 25)] +
                               [self.env_cfg.map_size / 10 for _ in range(0, 8)] +
                               [1000 for _ in range(0, 1)] +
                               [1 for _ in range(0, 3)])

        self.mask = np.array([0 for _ in range(0, 2)] +
                             [1 for _ in range(0, 1)] +
                             [1 for _ in range(0, 2)] +
                             [0 if i != 12 else 1 for i in range(0, 25)] +
                             [1, 1, 0, 0, 1, 1] +
                             [0 for _ in range(0, 2)] +
                             [0 for _ in range(0, 1)] +
                             [1 for _ in range(0, 3)])

        return
