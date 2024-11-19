# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 14:24:19 2024

@author: raymo
"""

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