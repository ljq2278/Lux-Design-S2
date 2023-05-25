import numpy as np
import torch
import copy


class Buffer:
    def __init__(self):
        self.uids = []
        self.states = []
        self.states_stat = []
        self.state_vals = []
        self.f_actions = []
        self.f_action_logprobs = []
        # self.f_action_masks = []
        self.u_actions = []
        self.u_action_logprobs = []
        # self.u_action_masks = []
        self.single_rewards = []
        self.rewards = []
        self.dones = []
        self.advantages = None

    def add_examples(self, uids, states, states_stat, state_vals, f_actions, f_action_logprobs, u_actions, u_action_logprobs, single_rewards, dones):
        self.uids += uids
        self.states += states
        self.states_stat += states_stat
        self.state_vals += state_vals
        self.f_actions += f_actions
        self.f_action_logprobs += f_action_logprobs
        # self.f_action_masks += f_action_masks
        self.u_actions += u_actions
        self.u_action_logprobs += u_action_logprobs
        # self.u_action_masks += u_action_masks
        self.single_rewards += single_rewards
        self.dones += dones
        return

    def transfer_reward(self, gamma):
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.single_rewards), reversed(self.dones)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (gamma * discounted_reward)
            self.rewards.insert(0, discounted_reward)
        return

    def transfer_reward_tdn(self, gamma, n):
        self.rewards = []
        last_end = -1
        for j, is_terminal in enumerate(self.dones):
            if is_terminal:
                tmp_single_rewards = np.array(self.single_rewards[last_end + 1:j + 1])
                cur_rewards = np.zeros(len(tmp_single_rewards))
                cur_state_vals = np.array(copy.deepcopy(self.state_vals[last_end + 1:j + 1]))
                for i in range(0, n):
                    cur_rewards = cur_rewards + gamma ** i * np.array(tmp_single_rewards[i:len(tmp_single_rewards)].tolist() + i * [0])
                    cur_state_vals = gamma * np.array(cur_state_vals[1:len(tmp_single_rewards)].tolist() + [0])
                self.rewards += (cur_rewards + cur_state_vals).tolist()
                last_end = j
        return

    def calc_advantage(self):
        self.advantages = (np.array(self.rewards) - np.array(self.state_vals)).tolist()

    def clear(self):
        del self.uids[:]
        del self.states[:]
        del self.states_stat[:]
        del self.state_vals[:]
        del self.f_actions[:]
        del self.f_action_logprobs[:]
        # del self.f_action_masks[:]
        del self.u_actions[:]
        del self.u_action_logprobs[:]
        # del self.u_action_masks[:]
        del self.single_rewards[:]
        del self.rewards[:]
        del self.dones[:]
