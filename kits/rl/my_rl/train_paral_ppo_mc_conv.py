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
tdn = 8
state_val_adv_debug = True
soft_update_tau = 0.2
gumbel_softmax_tau_online, gumbel_softmax_tau_train = 5, 5
lr = {
    'actor_u_lr': 0.00002,
    'actor_f_lr': 0.00001,
    'actor_u_mask_lr': 0.00002,
    'actor_f_mask_lr': 0.00001,
    'critic_lr': 0.00004,
    'encoder_lr': 0.00001,
    'decoder_lr': 0,
}
# actor_lr, critic_lr = 0.0005, 0.001
# encoder_lr, decoder_lr = 0.0001, 0.0000
v_loss_factor, f_loss_factor, u_loss_factor, entropy_loss_factor, ed_loss_factor = 1, 1, 1, 1, 0
l1_factor, l2_factor = 0.00, 0.00
eps_clip = 0.5
K_epochs = 5
episode_num = 3000000
gamma = 0.98
sub_proc_count = 4
exp = 'paral_ppo_share'
want_load_model = True
max_episode_length = 100
agent_debug = False
density_rwd = False
episode_start = 1
print_interv = 1
save_peri = 5
batch_size = 64
map_size = 24
os.environ['HOME'] = 'E:'
update_interv = 2
early_setup_strategy = 'resource'

dim_info = [ObsSpace(None).total_dims, ActSpaceFactory().f_dims, ActSpaceUnit().u_dims]  # obs and act dims
base_res_dir = os.environ['HOME'] + '/train_res/' + exp

writer = SummaryWriter(os.environ['HOME'] + '/logs/' + exp)
train_writer = SummaryWriter(os.environ['HOME'] + '/logs/' + exp + '_loss')


def sub_run(replay_queue: multiprocessing.Queue, param_queue: multiprocessing.Queue, process_id):
    env = gym.make(env_id, verbose=0, collect_stats=True, MAX_FACTORIES=3)
    env_cfg = env.env_cfg
    env_cfg.map_size = map_size
    env_cfg.max_episode_length = max_episode_length
    actTransfer = ActTransfer(env, env_cfg)
    obsTransfer = ObsTransfer(env, env_cfg)
    rwdTransfer = RwdTransfer(env, env_cfg, debug=False, density=density_rwd)
    agent_cont = 2
    online_agent = CentralOnlineAgent(dim_info[0], dim_info[1], dim_info[2], env_cfg, gumbel_softmax_tau=gumbel_softmax_tau_online)
    if want_load_model:
        new_params = param_queue.get()
        online_agent.update(new_params)
    globalAgents = [GlobalAgent('player_' + str(i), env_cfg, online_agent) for i in range(0, agent_cont)]
    globale_step = 0
    buffer = Buffer()
    tmp_buffer = {'player_0': []}  # record every datas
    for episode in range(episode_start, episode_num):
        np.random.seed()
        seed = np.random.randint(0, 100000000)
        seed = 88
        raw_obs = env.reset(seed=seed)
        done = {'player_0': False, 'player_1': False}
        episode_tmp_buffer = {'player_0': []}
        episode_reward = 0
        ################################ interact with the env for an episode ###################################
        while raw_obs['player_0']["real_env_steps"] < 0 or sum(done.values()) < len(done):
            globale_step += 1
            raw_action = {'player_0': {}, 'player_1': {}}
            if raw_obs['player_0']["real_env_steps"] < 0:
                for g_agent in globalAgents:
                    raw_action[g_agent.player] = g_agent.early_setup(env.get_state().env_steps, raw_obs[g_agent.player], strategy=early_setup_strategy)
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
                        else:
                            env.state.factories[p_id][f_id].cargo.water = 40
                            env.state.factories[p_id][f_id].cargo.metal += 2000  # int(1000 * np.random.random())
                            env.state.factories[p_id][f_id].power = 300000
                # print(raw_obs['player_0']["real_env_steps"], raw_action['player_0'])
                raw_next_obs, raw_reward, done, info = env.step(raw_action)
                # print(raw_obs['player_0']["real_env_steps"], env.state.stats['player_0'])
                obs, obs_stat = obsTransfer.raw_to_wrap(raw_obs['player_0'], env.state, max_episode_length - raw_obs['player_0']["real_env_steps"])
                last_stats = copy.deepcopy(env.state)
                raw_obs = raw_next_obs
            else:
                ############################### get action and raw_action factory ###############################
                f_action, u_action = {}, {}
                f_action_logprob, u_action_logprob = {}, {}
                state_val = {}
                for g_agent in [globalAgents[0]]:
                    f_action[g_agent.player], u_action[g_agent.player] = {}, {}
                    f_action_logprob[g_agent.player], u_action_logprob[g_agent.player] = {}, {}
                    state_val[g_agent.player], f_action[g_agent.player], f_action_logprob[g_agent.player], u_action[g_agent.player], u_action_logprob[g_agent.player], h \
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
                        last_stats.stats[g_agent.player],
                        env.state.stats[g_agent.player],
                        last_stats.factories[g_agent.player],
                        env.state.factories[g_agent.player],
                        last_stats.units[g_agent.player],
                        env.state.units[g_agent.player],
                        obs[g_agent.player],
                        next_obs[g_agent.player]
                    )
                    episode_reward += reward[g_agent.player]
                    ############################ record the simple data ################################
                    episode_tmp_buffer[g_agent.player].append([
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
                last_stats = copy.deepcopy(env.state)

        ############################### episode data record and print  #################################
        survive_step = raw_obs["player_0"]["real_env_steps"]
        message = f'episode {episode}, '
        message += f'episode reward: {episode_reward}, '
        message += f'survive step: {survive_step}'
        print(process_id, message)
        print(raw_obs["player_0"]["real_env_steps"], rwdTransfer.reward_collect)
        if process_id == 0:
            writer.add_scalars('rewards', rwdTransfer.reward_collect, episode)
        for k, v in rwdTransfer.reward_collect.items():
            rwdTransfer.reward_collect[k] = 0
        ############################## add train data #######################################################
        for k, v in episode_tmp_buffer.items():
            if episode_reward > -1000000:
                tmp_buffer[k] += v
        ##################### after a game, use MC the reward and get Advantage and tranport #####################
        if episode % update_interv == 0:
            # if len(tmp_buffer['player_0']) > 0:
            for p_id, behaviors in tmp_buffer.items():
                buffer.add_examples(*list(zip(*behaviors)))
            if tdn > 0:
                buffer.transfer_reward_tdn(gamma, tdn)
            else:
                buffer.transfer_reward(gamma)
            buffer.calc_advantage()
            if state_val_adv_debug:
                print('########################################################### state_vals advantages start ########################################################### ')
                print('state_vals: ', buffer.state_vals)
                print('advantages: ', buffer.advantages)
                print('########################################################### state_vals advantages end ########################################################### ')
            replay_queue.put(
                [buffer.states, buffer.states_stat, buffer.state_vals, buffer.f_actions, buffer.f_action_logprobs, buffer.u_actions, buffer.u_action_logprobs, buffer.rewards, buffer.dones,
                 buffer.advantages])
            new_params = param_queue.get()
            if soft_update_tau < 1:
                online_agent.soft_update(new_params, soft_update_tau)
            else:
                online_agent.update(new_params)
            buffer.clear()
            tmp_buffer = {'player_0': []}


def offline_learn(replay_queue: multiprocessing.Queue, param_queue_list, pid):
    env = gym.make(env_id, verbose=0, collect_stats=True, MAX_FACTORIES=2)
    env_cfg = env.env_cfg
    env_cfg.map_size = map_size
    offline_agent = CentralOfflineAgent(dim_info[0], dim_info[1], dim_info[2], env_cfg, lr, gumbel_softmax_tau=gumbel_softmax_tau_train, eps_clip=eps_clip, save_dir=base_res_dir)
    if want_load_model:
        offline_agent.load()
        for param_queue in param_queue_list:
            param_queue.put(offline_agent.policy.to('cpu').state_dict())
    online_agent_update_time = 0
    train_data = []
    while True:
        if replay_queue.qsize() == len(param_queue_list):
            while replay_queue.qsize() != 0:
                data = replay_queue.get()
                if len(data[0]) == 0:
                    print('data can not be null !')
                train_data.append(data)
            new_params = offline_agent.update_and_get_new_param2(train_data, K_epochs, batch_size, train_writer, online_agent_update_time, entropy_loss_factor,
                                                                 v_loss_factor, f_loss_factor, u_loss_factor, ed_loss_factor, l1_factor, l2_factor)
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
