import numpy as np
import torch


class Buffer:
    def __init__(self):
        self.uids = []
        self.states = []
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

    def add_examples(self, uids, states, state_vals, f_actions, f_action_logprobs, u_actions, u_action_logprobs,  single_rewards, dones):
        self.uids += uids
        self.states += states
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

    def calc_advantage(self):
        self.advantages = (np.array(self.rewards) - np.array(self.state_vals)).tolist()

    def clear(self):
        del self.uids[:]
        del self.states[:]
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
