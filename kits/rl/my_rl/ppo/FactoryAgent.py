from wrappers.obs_space_levels import ObsSpaceFactory
from wrappers.act_space_levels import ActSpaceFactory


class Factory_Agent():
    def __init__(self):
        pass

    def order_resource_pos(self, pf_info, ice_pos_list, ore_pos_list, rubble_pos_list=None):
        self.order_pos = {}
        for p_id, f_info in pf_info.items():
            for f_id, info in f_info.items():
                pos = info['pos']
                self.order_pos[f_id] = {}
                self.order_pos[f_id]['ice'] = sorted(ice_pos_list, key=lambda x: (x[0] - pos[0]) ** 2 + (x[1] - pos[1]) ** 2)
                self.order_pos[f_id]['ore'] = sorted(ore_pos_list, key=lambda x: (x[0] - pos[0]) ** 2 + (x[1] - pos[1]) ** 2)
                self.order_pos[f_id]['rubble'] = []

    def act(self, f_id, f_obs, uids):
        action = 3
        uid_target = {}
        if f_obs[ObsSpaceFactory.metal_dim_start] >= 100 and f_obs[ObsSpaceFactory.power_dim_start] >= 500:
            action = 1
        if f_obs[ObsSpaceFactory.water_dim_start] < 10000:
            for i, uid in enumerate(uids):
                uid_target[uid] = list(self.order_pos[f_id]['ice'][i])+list(f_obs[ObsSpaceFactory.pos_dim_start:ObsSpaceFactory.pos_dim_start + ObsSpaceFactory.pos_dim]) + ['ice']
        else:
            for i, uid in enumerate(uids):
                uid_target[uid] = list(self.order_pos[f_id]['ore'][i])+list(f_obs[ObsSpaceFactory.pos_dim_start:ObsSpaceFactory.pos_dim_start + ObsSpaceFactory.pos_dim]) + ['ore']
        return action, uid_target
