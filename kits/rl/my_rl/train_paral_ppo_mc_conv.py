"""
Implementation of RL agent. Note that luxai_s2 and stable_baselines3 are packages not available during the competition running (ATM)
"""
import multiprocessing
import copy
import os.path as osp
from torch.utils.tensorboard import SummaryWriter
import gym
import numpy as np
import os
import luxai_s2
from lux.config import EnvConfig
from wrappers.act_transfer import ActTransfer
from wrappers.rwd_transfer import RwdTransfer
from wrappers.obs_transfer import ObsTransfer
from agent.BaseAgent import EarlyRuleAgent

from wrappers.obs_space_conv import ObsSpace
from wrappers.act_space_conv import ActSpaceFactory, ActSpaceUnit
from ppo_conv.Buffer import Buffer
from ppo_conv.CentralAgent import CentralOnlineAgent, CentralOfflineAgent

from luxai_s2.map.position import Position


class GlobalAgent(EarlyRuleAgent):
    def __init__(self, player: str, env_cfg: EnvConfig, central_agent):
        EarlyRuleAgent.__init__(self, player, env_cfg)
        self.central_agent = central_agent


env_id = "LuxAI_S2-v0"
print_interv = 1
actor_lr = 0.0001
critic_lr = 0.0004
base_lr = 0.0001
eps_clip = 0.2
K_epochs = 1
episode_num = 3000000
gamma = 0.98
sub_proc_count = 5
exp = 'paral_ppo_share'
want_load_model = False
max_episode_length = 20
agent_debug = False
density_rwd = False
episode_start = 0
save_peri = 5
batch_size = 20
map_size = 32
os.environ['HOME'] = 'D:'
update_interv = 10

dim_info = [ObsSpace(None).total_dims, ActSpaceFactory().f_dims, ActSpaceUnit().u_dims]  # obs and act dims
base_res_dir = os.environ['HOME'] + '/train_res/' + exp

writer = SummaryWriter(os.environ['HOME'] + '/logs/' + exp)
train_writer = SummaryWriter(os.environ['HOME'] + '/logs/' + exp+'_loss')

def sub_run(replay_queue: multiprocessing.Queue, param_queue: multiprocessing.Queue, process_id):
    env = gym.make(env_id, verbose=0, collect_stats=True, MAX_FACTORIES=3)
    env_cfg = env.env_cfg
    env_cfg.map_size = map_size
    env_cfg.max_episode_length = max_episode_length
    actTransfer = ActTransfer(env, env_cfg)
    obsTransfer = ObsTransfer(env, env_cfg)
    rwdTransfer = RwdTransfer(env, env_cfg, debug=False, density=density_rwd)
    agent_cont = 2
    online_agent = CentralOnlineAgent(dim_info[0], dim_info[1], dim_info[2], env_cfg)
    if want_load_model:
        new_params = param_queue.get()
        online_agent.update(new_params)
    globalAgents = [GlobalAgent('player_' + str(i), env_cfg, online_agent) for i in range(0, agent_cont)]
    globale_step = 0
    sum_rwd = 0
    survive_step = 0
    buffer = Buffer()
    tmp_buffer = {}  # record every datas
    for episode in range(episode_start, episode_num):
        np.random.seed()
        seed = np.random.randint(0, 100000000)
        raw_obs = env.reset(seed=seed)
        done = {'player_0': False, 'player_1': False}
        ################################ interact with the env for an episode ###################################
        while raw_obs['player_0']["real_env_steps"] < 0 or sum(done.values()) < len(done):
            globale_step += 1
            raw_action = {'player_0': {}, 'player_1': {}}
            if raw_obs['player_0']["real_env_steps"] < 0:
                for g_agent in globalAgents:
                    raw_action[g_agent.player] = g_agent.early_setup(env.get_state().env_steps, raw_obs[g_agent.player])
                raw_next_obs, raw_reward, done, info = env.step(raw_action)
                raw_obs = raw_next_obs
            elif raw_obs['player_0']["real_env_steps"] == 0:
                for p_id, fp_info in env.state.factories.items():
                    for f_id in fp_info.keys():
                        raw_action[p_id][f_id] = 1
                        if p_id == 'player_1':
                            # set factories to have 1000 water to check the ore dig ability
                            env.state.factories[p_id][f_id].cargo.water = 200
                            # env.state.factories[p_id][f_id].cargo.metal = 200
                            # env.state.factories[p_id][f_id].power = 300000
                # print(raw_obs['player_0']["real_env_steps"], raw_action['player_0'])
                raw_next_obs, raw_reward, done, info = env.step(raw_action)
                # print(raw_obs['player_0']["real_env_steps"], env.state.stats['player_0'])
                obs, obs_stat = obsTransfer.raw_to_wrap(raw_obs['player_0'], env.state, max_episode_length - raw_obs['player_0']["real_env_steps"])
                last_stats = copy.deepcopy(env.state.stats)
                raw_obs = raw_next_obs
            else:
                ############################### get action and raw_action factory ###############################
                f_action, u_action = {}, {}
                f_action_logprob, u_action_logprob = {}, {}
                state_val = {}
                for g_agent in [globalAgents[0]]:
                    f_action[g_agent.player], u_action[g_agent.player] = {}, {}
                    f_action_logprob[g_agent.player], u_action_logprob[g_agent.player] = {}, {}
                    state_val[g_agent.player], f_action[g_agent.player], f_action_logprob[g_agent.player], u_action[g_agent.player], u_action_logprob[g_agent.player] \
                        = online_agent.policy.act(np.array([obs[g_agent.player]]), np.array([obs_stat[g_agent.player]]))
                    state_val[g_agent.player], f_action[g_agent.player], f_action_logprob[g_agent.player], u_action[g_agent.player], u_action_logprob[g_agent.player] \
                        = state_val[g_agent.player][0][0], f_action[g_agent.player][0], f_action_logprob[g_agent.player][0], u_action[g_agent.player][0], u_action_logprob[g_agent.player][0]
                    raw_action[g_agent.player] = actTransfer.wrap_to_raw(
                        f_action[g_agent.player], u_action[g_agent.player], env.state.factories[g_agent.player], env.state.units[g_agent.player])
                ############################### get action to env result ###############################
                # print(raw_obs['player_0']["real_env_steps"], raw_action['player_0'])
                raw_next_obs, raw_reward, done, info = env.step(raw_action)
                # print(raw_obs['player_0']["real_env_steps"], env.state.stats['player_0'])
                ############################### get next obs factory ######################################
                next_obs, next_obs_stat = obsTransfer.raw_to_wrap(raw_next_obs['player_0'], env.state, max_episode_length - raw_obs['player_0']["real_env_steps"])
                ############################### get custom reward factory ######################################
                reward = {}
                for g_agent in [globalAgents[0]]:
                    reward[g_agent.player], _ = rwdTransfer.raw_to_wrap(
                        raw_reward[g_agent.player],
                        done[g_agent.player],
                        last_stats[g_agent.player],
                        env.state.stats[g_agent.player]
                    )
                    sum_rwd += reward[g_agent.player]
                    ############################ record the simple data ################################
                    if g_agent.player not in tmp_buffer.keys():
                        tmp_buffer[g_agent.player] = []
                    tmp_buffer[g_agent.player].append([
                        g_agent.player,
                        obs[g_agent.player],
                        obs_stat[g_agent.player],
                        state_val[g_agent.player],
                        f_action[g_agent.player],
                        f_action_logprob[g_agent.player],
                        u_action[g_agent.player],
                        u_action_logprob[g_agent.player],
                        reward[g_agent.player],
                        done[g_agent.player]
                    ])
                ############################### prepare to the next step #################################
                raw_obs = raw_next_obs
                obs = next_obs
                obs_stat = next_obs_stat
                last_stats = copy.deepcopy(env.state.stats)

        ############################### episode data record  #################################
        survive_step += raw_obs["player_0"]["real_env_steps"]
        ##################### after a game, use MC the reward and get Advantage and tranport #####################
        if episode % update_interv == update_interv - 1:
            for p_id, behaviors in tmp_buffer.items():
                buffer.add_examples(*list(zip(*behaviors)))
            buffer.transfer_reward(gamma)
            buffer.calc_advantage()
            replay_queue.put(
                [buffer.states, buffer.states_stat, buffer.state_vals, buffer.f_actions, buffer.f_action_logprobs, buffer.u_actions, buffer.u_action_logprobs, buffer.rewards, buffer.dones,
                 buffer.advantages])
            new_params = param_queue.get()
            online_agent.update(new_params)
            buffer.clear()
            tmp_buffer.clear()

        ########################################### episode finishes  ########################################
        if episode % print_interv == 0:  # print info every 100 g_step
            message = f'episode {episode}, '
            message += f'avg episode reward: {sum_rwd / print_interv}, '
            message += f'avg survive step: {survive_step / print_interv}'
            print(message)
            print(raw_obs["player_0"]["real_env_steps"], rwdTransfer.reward_collect)
            sum_rwd = 0
            survive_step = 0
            if process_id == 0:
                writer.add_scalars('rewards', rwdTransfer.reward_collect, episode)
            for k, v in rwdTransfer.reward_collect.items():
                rwdTransfer.reward_collect[k] = 0


def offline_learn(replay_queue: multiprocessing.Queue, param_queue_list, pid):
    env = gym.make(env_id, verbose=0, collect_stats=True, MAX_FACTORIES=2)
    env_cfg = env.env_cfg
    env_cfg.map_size = map_size
    offline_agent = CentralOfflineAgent(dim_info[0], dim_info[1], dim_info[2], env_cfg, actor_lr, critic_lr, base_lr, eps_clip, base_res_dir)
    if want_load_model:
        offline_agent.load()
        for param_queue in param_queue_list:
            param_queue.put(offline_agent.policy.state_dict())
    online_agent_update_time = 0
    train_data = []
    while True:
        if replay_queue.qsize() == len(param_queue_list):
            while replay_queue.qsize() != 0:
                data = replay_queue.get()
                if len(data[0]) == 0:
                    print('data can not be null !')
                train_data.append(data)
            new_params = offline_agent.update_and_get_new_param2(train_data, K_epochs, batch_size, train_writer, online_agent_update_time)
            online_agent_update_time += 1
            train_data.clear()
            for param_queue in param_queue_list:
                param_queue.put(new_params)
            if online_agent_update_time % save_peri == 0:
                print('save model, online agent update time ', online_agent_update_time)
                offline_agent.save()


if __name__ == "__main__":
    replay_queue = multiprocessing.Queue(maxsize=0)
    param_queue_list = [multiprocessing.Queue(maxsize=0) for _ in range(0, sub_proc_count)]

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
