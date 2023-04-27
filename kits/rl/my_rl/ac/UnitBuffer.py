import numpy as np
import torch


class Buffer:
    """replay buffer for each agent"""

    def __init__(self, capacity, obs_dim, act_dim, device):
        self.obs = []
        self.action = []
        self.reward = []
        self.next_obs = []
        self.done = []

    def add(self, obs, action, reward, next_obs, done):
        """ add an experience to the memory """
        self.obs.append(obs)
        self.action.append(action)
        self.reward.append(reward)
        self.next_obs.append(next_obs)
        self.done.append(done)

    def get_replays(self):
        # retrieve data, Note that the data stored is ndarray
        return self.obs, self.action, self.reward, self.next_obs, self.done

    def clear(self):
        self.obs.clear()
        self.action.clear()
        self.reward.clear()
        self.next_obs.clear()
        self.done.clear()

    def __len__(self):
        return len(self.done)
