import numpy as np
import torch


class Buffer:
    """replay buffer for each agent"""

    def __init__(self, capacity, obs_dim, act_dim, device):
        self.capacity = capacity

        self.obs = np.zeros((capacity, obs_dim))
        self.action = np.zeros((capacity, 1))
        self.reward = np.zeros(capacity)
        self.next_obs = np.zeros((capacity, obs_dim))
        self.done = np.zeros(capacity, dtype=bool)

        self.size = 0

        self.device = device

    def add(self, obs, action, reward, next_obs, done):
        """ add an experience to the memory """
        if self.size < self.capacity:
            self.obs[self.size] = obs
            self.action[self.size] = action
            self.reward[self.size] = reward
            self.next_obs[self.size] = next_obs
            self.done[self.size] = done
            self.size += 1
        else:
            print("full of buff !")

    def get(self):
        # retrieve data, Note that the data stored is ndarray
        if self.size != self.capacity:
            raise Exception("batch no right !")
        else:
            self.size = 0
            return self.obs, self.action, self.reward, self.next_obs, self.done

    def __len__(self):
        return self.size
