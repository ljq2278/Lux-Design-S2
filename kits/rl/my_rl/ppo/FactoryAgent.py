from wrappers.obs_space_levels import ObsSpaceFactory
from wrappers.act_space_levels import ActSpaceFactory


class Factory_Agent():
    def __init__(self):
        pass

    def order_resource_pos(self, pf_info, ice_pos_list, ore_pos_list, rubble_pos_list=None):
        self.order_pos = {}
        for p_id, f_info in pf_info.items():
            self.order_pos[p_id] = {}
            for f_id, info in f_info.items():
                pos = info['pos']
                self.order_pos[p_id][f_id] = {}
                self.order_pos[p_id][f_id]['ice'] = sorted(ice_pos_list, key=lambda x: (x[0] - pos[0]) ** 2 + (x[1] - pos[1]) ** 2)
                self.order_pos[p_id][f_id]['ore'] = sorted(ore_pos_list, key=lambda x: (x[0] - pos[0]) ** 2 + (x[1] - pos[1]) ** 2)
                self.order_pos[p_id][f_id]['rubble'] = sorted(rubble_pos_list, key=lambda x: (x[0] - pos[0]) ** 2 + (x[1] - pos[1]) ** 2)
                self.order_pos[p_id][f_id]['pos'] = pos

    def update_rubble_pos(self, pf_info, rubble_pos_list):
        for p_id, f_info in pf_info.items():
            for f_id, info in f_info.items():
                pos = info['pos']
                self.order_pos[p_id][f_id]['rubble'] = sorted(rubble_pos_list, key=lambda x: (x[0] - pos[0]) ** 2 + (x[1] - pos[1]) ** 2)

    def act(self, f_obs, step):
        action = 3
        task = 'ice'
        ###################################### the action choice ########################################
        if f_obs[ObsSpaceFactory.metal_dim_start] >= 100 and f_obs[ObsSpaceFactory.power_dim_start] >= 500 and step % 5 == 0:
            action = 1
        elif step > 800:
            action = 2
        ######################################### the demand ##################################################
        if f_obs[ObsSpaceFactory.water_dim_start] < 500:
            task = 'ice'
            # for i, uid in enumerate(uids):
            #     uid_target[uid] = list(self.order_pos[f_id]['ice'][i])+list(f_obs[ObsSpaceFactory.pos_dim_start:ObsSpaceFactory.pos_dim_start + ObsSpaceFactory.pos_dim]) + ['ice']
        elif f_obs[ObsSpaceFactory.water_dim_start] > 600:
            task = 'rubble'
            # for i, uid in enumerate(uids):
            #     uid_target[uid] = list(self.order_pos[f_id]['ore'][i])+list(f_obs[ObsSpaceFactory.pos_dim_start:ObsSpaceFactory.pos_dim_start + ObsSpaceFactory.pos_dim]) + ['ore']
        else:
            task = 'ore'
        return action, task
