import numpy as np
import torch


class Buffer:
    def __init__(self):
        self.uids = []
        self.actions = []
        self.states = []
        self.action_logprobs = []
        self.single_rewards = []
        self.rewards = []
        self.state_vals = []
        self.dones = []
        self.advantages = None



    def add_examples(self, uids, actions, states, action_logprobs, single_rewards, state_vals, dones):
        self.uids += uids
        self.states += states
        self.actions += actions
        self.action_logprobs += action_logprobs
        self.state_vals += state_vals
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

    def transfer_reward_tdn(self, gamma):
        valid_indexes = []
        for i, (reward, is_terminal, state_val, uid) in enumerate(zip(reversed(self.single_rewards), reversed(self.dones), reversed(self.state_vals), reversed(self.uids))):
            if i == 0:
                discounted_reward = state_val
            elif uid != last_uid:
                discounted_reward = state_val
            else:
                if is_terminal:
                    discounted_reward = 0
                discounted_reward = reward + (gamma * discounted_reward)
                self.rewards.insert(0, discounted_reward)
                valid_indexes.insert(0, len(self.dones)-1-i)
            last_uid = uid
        self.uids = [self.uids[j] for j in valid_indexes]
        self.states = [self.states[j] for j in valid_indexes]
        self.actions = [self.actions[j] for j in valid_indexes]
        self.action_logprobs = [self.action_logprobs[j] for j in valid_indexes]
        self.state_vals = [self.state_vals[j] for j in valid_indexes]
        self.single_rewards = [self.single_rewards[j] for j in valid_indexes]
        self.dones =[self.dones[j] for j in valid_indexes]
        return

    def calc_advantage(self):
        self.advantages = (np.array(self.rewards) - np.array(self.state_vals)).tolist()

    def clear(self):
        del self.uids[:]
        del self.actions[:]
        del self.states[:]
        del self.action_logprobs[:]
        del self.single_rewards[:]
        del self.rewards[:]
        del self.state_vals[:]
        del self.dones[:]
