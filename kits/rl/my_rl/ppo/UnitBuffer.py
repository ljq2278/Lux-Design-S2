import numpy as np
import torch


class Buffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.action_logprobs = []
        self.single_rewards = []
        self.rewards = []
        self.state_vals = []
        self.dones = []
        self.advantages = None

    def add_example(self, action, state, action_logprob, single_reward, state_val, done):
        self.states.append(state)
        self.actions.append(action)
        self.action_logprobs.append(action_logprob)
        self.state_vals.append(state_val)
        self.single_rewards.append(single_reward)
        self.dones.append(done)
        return

    def add_examples(self, actions, states, action_logprobs, single_rewards, state_vals, dones):
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

    def calc_advantage(self):
        self.advantages = (np.array(self.rewards) - np.array(self.state_vals)).tolist()

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.action_logprobs[:]
        del self.single_rewards[:]
        del self.rewards[:]
        del self.state_vals[:]
        del self.dones[:]
