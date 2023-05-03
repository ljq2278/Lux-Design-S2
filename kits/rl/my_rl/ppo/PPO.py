import logging
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F

from ac.UnitAgent import UnitAgent
from ac.UnitBuffer import Buffer


class PPO:
    """A MADDPG(Multi Agent Deep Deterministic Policy Gradient) agent"""

    def __init__(self, unit_agent, factory_agent=None):
        self.unit_agent = unit_agent
        self.factory_agent = factory_agent
