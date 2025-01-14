# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 14:24:19 2024

@author: raymo
"""

import numpy as np
import time

class Profiler:
    
    def __init__(self):
        self.last = time.time()
        self.times = {}
    
    def log(self, time_idx):
        now = time.time()
        tt = now - self.last
        self.last = now
        if time_idx not in self.times:
            self.times[time_idx] = 0
        self.times[time_idx] += tt
    
    def display(self):
        for k, v in self.times.items():
            print('Cum time {}: {:.3f}'.format(k, v))
        tot_time = sum(self.times.values())
        print('Total time: {:.3f}'.format(tot_time))

def idx_to_rowcol(idx):
    
    row = idx // 3
    col = idx % 3
    
    return row, col

def rowcol_to_idx(row, col):
    
    return 3 * row + col

def onehot_mask(list_of_indexes, length):
    
    oh_list = [False] * length
    for idx in list_of_indexes:
        oh_list[idx] = True
    
    return oh_list

def moving_average(a, n=1000):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n