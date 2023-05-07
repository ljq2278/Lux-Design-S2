"""
Implementation of RL agent. Note that luxai_s2 and stable_baselines3 are packages not available during the competition running (ATM)
"""
import multiprocessing
import copy
import os.path as osp

import gym
import numpy as np
import luxai_s2
import os
from lux.config import EnvConfig
from wrappers.ma_act_transor_levels import MaActTransorFactory, MaActTransorUnit
from wrappers.ma_rwd_transor_levels import MaRwdTransorFactory, MaRwdTransorUnit
from wrappers.ma_obs_transor_levels import MaObsTransorFactory, MaObsTransorUnit
from agent.BaseAgent import EarlyRuleAgent
from ppo.PPO import PPO
from ppo.UnitAgent import PPO_Offline_Agent
from ppo.UnitAgent import PPO_Online_Agent
from ppo.UnitBuffer import Buffer
from ppo.FactoryAgent import Factory_Agent
import cv2

class GlobalAgent(EarlyRuleAgent, PPO):
    def __init__(self, player: str, env_cfg: EnvConfig, unit_agent, factory_agent):
        EarlyRuleAgent.__init__(self, player, env_cfg)
        PPO.__init__(self, unit_agent, factory_agent)


env_id = "LuxAI_S2-v0"
print_interv = 10
actor_lr = 0.0004
critic_lr = 0.001
eps_clip = 0.2
K_epochs = 0
episode_num = 3000000
gamma = 0.98
sub_proc_count = 1
exp = 'paral_ppo2'
want_load_model = True
max_episode_length = 1000
agent_debug = False
density_rwd = True

dim_info_unit = [MaObsTransorUnit.total_dims, MaActTransorUnit.total_act_dims]  # obs and act dims
base_res_dir = os.environ['HOME'] + '/train_res/' + exp


def sub_run(replay_queue: multiprocessing.Queue, param_queue: multiprocessing.Queue, p_id):
    # unit_agent = unit_agent.value
    env = gym.make(env_id, verbose=0, collect_stats=True, MAX_FACTORIES=2)
    env_cfg = env.env_cfg
    env_cfg.max_episode_length = max_episode_length
    maActTransorUnit = MaActTransorUnit(env, env_cfg)
    maActTransorFactory = MaActTransorFactory(env, env_cfg)
    maObsTransorUnit = MaObsTransorUnit(env, env_cfg, if_mask=True)
    maObsTransorFactory = MaObsTransorFactory(env, env_cfg, if_mask=True)
    maRwdTransorUnit = MaRwdTransorUnit(env, env_cfg, debug=True, density=density_rwd)
    maRwdTransorFactory = MaRwdTransorFactory(env, env_cfg, debug=True, density=density_rwd)
    agent_cont = 2
    unit_online_agent = PPO_Online_Agent(dim_info_unit[0], dim_info_unit[1], env_cfg)
    factory_agent = Factory_Agent()
    if want_load_model:
        new_params = param_queue.get()
        unit_online_agent.update(new_params)
    globalAgents = [GlobalAgent('player_' + str(i), env_cfg, unit_online_agent, factory_agent) for i in range(0, agent_cont)]
    globale_step = 0
    sum_rwd = 0
    survive_step = 0
    unit_buffer = Buffer()
    tmp_buffer_unit = {}  # record every unit datas
    for episode in range(1):
        np.random.seed()
        seed = np.random.randint(0, 100000000)
        raw_obs = env.reset(seed=seed)
        obs_unit = maObsTransorUnit.sg_to_ma(raw_obs['player_0'], None)
        obs_factory = maObsTransorFactory.sg_to_ma(raw_obs['player_0'])
        done = {'player_0': False, 'player_1': False}
        imgs = []
        ################################ interact with the env for an episode ###################################
        while raw_obs['player_0']["real_env_steps"] < 0 or sum(done.values()) < len(done):
            if raw_obs['player_0']["real_env_steps"] < 0:
                raw_action = {}
                for g_agent in globalAgents:
                    raw_action[g_agent.player] = g_agent.early_setup(env.get_state().env_steps, raw_obs[g_agent.player])
                raw_next_obs, raw_reward, done, info = env.step(raw_action)
                raw_obs = raw_next_obs
            else:
                rubble_locs = np.argwhere(raw_obs['player_0']["board"]["rubble"] > 0)
                if raw_obs['player_0']["real_env_steps"] == 0:
                    for p_id, fp_info in env.state.factories.items():
                        for f_id in fp_info.keys():
                            # set factories to have 1000 water to check the ore dig ability
                            env.state.factories[p_id][f_id].cargo.water = 300
                            env.state.factories[p_id][f_id].cargo.metal = 500
                            env.state.factories[p_id][f_id].power = 10000
                    obs_factory = maObsTransorFactory.sg_to_ma(raw_obs['player_0'])
                    ice_locs = np.argwhere(raw_obs['player_0']["board"]["ice"] == 1)
                    ore_locs = np.argwhere(raw_obs['player_0']["board"]["ore"] == 1)
                    factory_agent.order_resource_pos(raw_obs['player_0']['factories'], ice_locs, ore_locs, rubble_locs)
                else:
                    factory_agent.update_rubble_pos(raw_obs['player_0']['factories'], rubble_locs)
                globale_step += 1
                raw_action = {}
                ############################### get action and raw_action factory ###############################
                action_factory = {}
                raw_action_factory = {}
                factory_task = {}
                for g_agent in globalAgents:
                    action_factory[g_agent.player] = {}
                    factory_task[g_agent.player] = {}
                    for f_id, f_obs in obs_factory[g_agent.player].items():
                        action_factory[g_agent.player][f_id], factory_task[g_agent.player][f_id] = factory_agent.act(f_obs, raw_obs['player_0']["real_env_steps"])
                    raw_action_factory[g_agent.player] = maActTransorFactory.ma_to_sg(action_factory[g_agent.player], raw_obs[g_agent.player], g_agent.player)
                ################################ change the unit obs #############################################################################
                obs_unit = maObsTransorUnit.change_uobs_with_order(obs_unit, factory_task, factory_agent.order_pos)
                ############################### get action and raw_action unit ###################################################################
                action_unit = {}
                action_logprob_unit = {}
                state_val_unit = {}
                raw_action_unit = {}
                for g_agent in globalAgents:
                    action_unit[g_agent.player] = {}
                    action_logprob_unit[g_agent.player] = {}
                    state_val_unit[g_agent.player] = {}
                    for u_id, u_obs in obs_unit[g_agent.player].items():
                        a, b, c = unit_online_agent.policy.act([u_obs])
                        action_unit[g_agent.player][u_id], action_logprob_unit[g_agent.player][u_id], state_val_unit[g_agent.player][u_id] = a[0], b[0], c[0][0]
                    raw_action_unit[g_agent.player] = maActTransorUnit.ma_to_sg(action_unit[g_agent.player], obs_unit[g_agent.player])
                ############################### get action to env result ###############################
                for p_id, p_info in raw_action_factory.items():
                    raw_action[p_id] = {}
                    for f_id, f_info in p_info.items():
                        raw_action[p_id][f_id] = f_info
                    for u_id, u_info in raw_action_unit[p_id].items():
                        raw_action[p_id][u_id] = u_info
                raw_next_obs, raw_reward, done, info = env.step(raw_action)
                imgs += [env.render("rgb_array", width=640, height=640)]
                ############################### get next obs factory ######################################
                next_obs_factory = maObsTransorFactory.sg_to_ma(raw_next_obs['player_0'])
                ############################### get custom reward factory ######################################
                reward_factory = {}
                for g_agent in globalAgents:
                    reward_factory[g_agent.player] = maRwdTransorFactory.sg_to_ma(
                        raw_reward[g_agent.player],
                        action_factory[g_agent.player],
                        obs_factory[g_agent.player],
                        next_obs_factory[g_agent.player],
                        done[g_agent.player]
                    )
                    for f_id, f_info in obs_factory[g_agent.player].items():
                        if f_id not in next_obs_factory[g_agent.player].keys() or done[g_agent.player]:
                            d_factory = True
                        else:
                            d_factory = False
                        sum_rwd += reward_factory[g_agent.player][f_id]
                        ############################ record the simple data ################################
                        # if we want do some factory action samples record
                ############################### get next obs unit ######################################
                next_obs_unit = maObsTransorUnit.sg_to_ma(raw_next_obs['player_0'], obs_unit)
                ################################ change the unit obs #############################################################################
                next_obs_unit = maObsTransorUnit.change_uobs_with_order(next_obs_unit, factory_task, factory_agent.order_pos)
                ############################### get custom reward unit ######################################
                reward_unit = {}
                for g_agent in globalAgents:
                    reward_unit[g_agent.player], next_obs_unit[g_agent.player] = maRwdTransorUnit.sg_to_ma(
                        raw_reward[g_agent.player],
                        action_unit[g_agent.player],
                        obs_unit[g_agent.player],
                        next_obs_unit[g_agent.player],
                        done[g_agent.player]
                    )
                    for u_id, u_info in obs_unit[g_agent.player].items():
                        if u_id not in next_obs_unit[g_agent.player].keys() or done[g_agent.player]:
                            d_unit = True
                        else:
                            d_unit = False
                        sum_rwd += reward_unit[g_agent.player][u_id]
                        ############################ record the simple data ################################
                        if u_id not in tmp_buffer_unit.keys():
                            tmp_buffer_unit[u_id] = []
                        tmp_buffer_unit[u_id].append([
                            u_id,
                            action_unit[g_agent.player][u_id],
                            obs_unit[g_agent.player][u_id],
                            action_logprob_unit[g_agent.player][u_id],
                            reward_unit[g_agent.player][u_id],
                            state_val_unit[g_agent.player][u_id],
                            d_unit
                        ])
                ############################### prepare to the next step #################################
                raw_obs = raw_next_obs
                obs_factory = next_obs_factory
                obs_unit = next_obs_unit
        ############################### episode data record  #################################
        survive_step += raw_obs["player_0"]["real_env_steps"]
        for img in imgs:
            cv2.imshow("Window", img)
            cv2.moveWindow('Window', 100, 100)
            cv2.waitKey(800)
            cv2.destroyAllWindows()
            # break
        ##################### after a game, use MC the reward and get Advantage and tranport #####################
        for u_id, behaviors in tmp_buffer_unit.items():
            unit_buffer.add_examples(*list(zip(*behaviors)))
        unit_buffer.transfer_reward(gamma)
        unit_buffer.calc_advantage()
        replay_queue.put(
            [unit_buffer.states, unit_buffer.actions, unit_buffer.action_logprobs,
             unit_buffer.state_vals, unit_buffer.rewards, unit_buffer.dones, unit_buffer.advantages])
        new_params = param_queue.get()
        unit_online_agent.update(new_params)
        unit_buffer.clear()
        tmp_buffer_unit.clear()

        ########################################### episode finishes  ########################################
        if episode % print_interv == 0:  # print info every 100 g_step
            message = f'episode {episode}, '
            message += f'avg episode reward: {sum_rwd / print_interv}, '
            message += f'avg survive step: {survive_step / print_interv}'
            print(message)
            print(raw_obs["player_0"]["real_env_steps"], maRwdTransorUnit.reward_collect)
            print(raw_obs["player_0"]["real_env_steps"], maRwdTransorFactory.reward_collect)
            sum_rwd = 0
            survive_step = 0
            for k, v in maRwdTransorUnit.reward_collect.items():
                maRwdTransorUnit.reward_collect[k] = 0
            for k, v in maRwdTransorFactory.reward_collect.items():
                maRwdTransorFactory.reward_collect[k] = 0


def offline_learn(replay_queue: multiprocessing.Queue, param_queue_list, pid):
    env = gym.make(env_id, verbose=0, collect_stats=True, MAX_FACTORIES=2)
    env_cfg = env.env_cfg
    ppo_offline_agent = PPO_Offline_Agent(dim_info_unit[0], dim_info_unit[1], env_cfg, actor_lr, critic_lr, eps_clip, base_res_dir)
    if want_load_model:
        ppo_offline_agent.load()
        for param_queue in param_queue_list:
            param_queue.put(ppo_offline_agent.policy.state_dict())
    online_agent_update_time = 0
    train_data = []
    while True:
        if replay_queue.qsize() == len(param_queue_list):
            while replay_queue.qsize() != 0:
                data = replay_queue.get()
                train_data.append(data)
            new_params = ppo_offline_agent.update_and_get_new_param(train_data, K_epochs)
            online_agent_update_time += 1
            train_data.clear()
            for param_queue in param_queue_list:
                param_queue.put(new_params)

if __name__ == "__main__":
    replay_queue = multiprocessing.Queue()
    param_queue_list = [multiprocessing.Queue() for _ in range(0, sub_proc_count)]

    processes = []

    # Create some sub-processes that use the shared module
    for i in range(sub_proc_count):
        process = multiprocessing.Process(target=sub_run, args=(replay_queue, param_queue_list[i], i))
        process.start()
        processes.append(process)

    offline_learn_process = multiprocessing.Process(target=offline_learn, args=(replay_queue, param_queue_list, -1))
    offline_learn_process.start()
    processes.append(offline_learn_process)

    # Wait for the sub-processes to finish
    for process in processes:
        process.join()

    print('end .............')
    # Print the predictions
    # print(predictions)
