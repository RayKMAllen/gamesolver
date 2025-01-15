# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 15:13:14 2025

@author: raymo
"""

import numpy as np
import random


class Agent:
    
    def __init__(self, gamma=0.95):
        self.gamma = gamma          # reward discount per step
        # self.epsilon = epsilon      # exploration control
        pass
        

class RandomAgent(Agent):
    
    def __init__(self):
        super().__init__()
        self.history = None
        
    def choose(self, game, player, possible_moves, encoded_state=None):
        
        # move = np.random.choice(possible_moves)
        move = random.choice(possible_moves)
        return move