import numpy as np
import random
from collections import deque


class Memory:
    def __init__(self, capacity):
        self.data = deque(maxlen=capacity)
        self.pointer = 0

    def remember(self, state, action, reward, state_next, done):
        experience = (state, action, reward, state_next, done)
        self.data.append(experience)
        if self.pointer < len(self.data):
            self.pointer += 1

    def sample(self, batch, agents=1):
        """
        If 1 agent, assumes no data about other agents.
        If 2+ agents, assumes data contains all agent data.
        """
        if agents == 1:
            states = np.array([self.data[i][0] for i in batch])
            actions = np.array([self.data[i][1] for i in batch])
            states_next = np.array([self.data[i][3] for i in batch])
        else:
            states = []
            actions = []
            states_next = []
            for a in range(agents):
                states.append(np.array([self.data[i][0][a] for i in batch]))
                actions.append(np.array([self.data[i][1][a] for i in batch]))
                states_next.append(np.array([self.data[i][3][a]
                                             for i in batch]))

        rewards = np.array([self.data[i][2] for i in batch])
        dones = np.array([self.data[i][4] for i in batch])

        return states, actions, rewards, states_next, dones

    def __str__(self):
        memory_state = ""
        for s, a, r, sn, done in self.data:
            if isinstance(s, list):
                # probably agents 2+
                for i in s:
                    memory_state += "{},".format(i.shape)
                memory_state += ";"
            else:
                memory_state += "{};".format(s.shape)
        return memory_state
