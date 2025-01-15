# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 12:45:16 2024

@author: raymo
"""

import numpy as np
import random
import torch
import torch_directml
from torch import tensor
from torch import nn
from torch.nn import Conv1d, Conv2d, Conv3d
from collections import deque
from matplotlib import pyplot as plt
import time
import asyncio
import nest_asyncio

from utils import Profiler, moving_average
from history import History
from game import TicTacToe, TTTPlayer
from agent import Agent, RandomAgent

nest_asyncio.apply()



class DLAgent(Agent):

    def __init__(self, model, gamma=0.95, per=False):
        self.gamma = gamma
        self.per = per
        super().__init__(gamma=gamma)
        self.history = History(per=per)
        self.model = model
        self.losses = []
        
    def choose(self, game, player, possible_moves, encoded_state=None):

        # self.model.eval()
        # profiler.log(10)
        
        if encoded_state is None:
            encoded_state = game.encode_state(player)
        # profiler.log(11)
        batched_state = encoded_state.unsqueeze(0)
        # profiler.log(12)
        output = self.model(batched_state)
        # profiler.log(13)
        # output = self.model(batched_state.to(dml))
        
        # todo various choice algos (prob distribution, eps-greedy etc.)
        if random.random() < player.epsilon:
            move = np.random.choice(possible_moves)
            # profiler.log(14)
            return move
        else:
            # greedy
            possible_moves_outputs = output[0][possible_moves]
            # profiler.log(15)
            best_move_idx = possible_moves_outputs.argmax()
            # profiler.log(16)
            best_move = possible_moves[best_move_idx]
            # profiler.log(17)
            
            return best_move

    def train_batch(self, batch_size=16):
        
        self.model.train()
        
        if self.per:
            (batch_states, 
             batch_actions, 
             batch_next_states, 
             batch_nspas, 
             batch_fr_mix, 
             batch_final_rewards,
             weights,    # TODO not currently using, needed for double-Q (see link in sumtree.py)
             tree_idxs,
             ) = self.history.yield_batch(batch_size)
        else:
            (batch_states, 
             batch_actions, 
             batch_next_states, 
             batch_nspas, 
             batch_fr_mix, 
             batch_final_rewards
             ) = self.history.yield_batch(batch_size)

                
        outputs = self.model(batch_states)
        state_action_outputs = outputs.gather(1, batch_actions.unsqueeze(-1)).squeeze()
        
        with torch.no_grad():
            next_state_outputs = self.model(batch_next_states)
            
            # apply mask
            next_state_outputs[~batch_nspas] = -np.inf
            
            # get value using max
            next_state_values = next_state_outputs.max(axis=1)
            next_state_values = next_state_values.values
        
        targets = batch_fr_mix * batch_final_rewards + (1 - batch_fr_mix) * next_state_values
        
        # for PER
        td_error = torch.abs(state_action_outputs - targets).detach()
        
        loss = self.model.loss(state_action_outputs, targets)
        self.model.optimizer.zero_grad()
        loss.backward()
        
        loss_scalar = loss.item()
        # print('Loss: {:.5f}'.format(loss_scalar))
        self.losses.append(loss_scalar)
        
        self.model.optimizer.step()
        
        if self.per:
            # update PER priorities
            self.history.update_priorities(tree_idxs, td_error)

        self.model.eval()
    
    def train_n_batches(self, n, batch_size=16):
        
        for i in range(n):
            self.train_batch(batch_size=batch_size)
        
                

class TTTConvModel(nn.Module):

    # def __init__(self, layer_1_size, layer_2_size, linear_size, lr=1e-3, padding=True):
    def __init__(self, input_channels, input_dimensions, num_actions, conv_layer_sizes, linear_sizes, kernel_size=3, stride=1, lr=1e-3, padding=True):
        super().__init__()
        # self.layer_1_size = layer_1_size
        # self.layer_2_size = layer_2_size
        # self.linear_size = linear_size
        self.input_channels = input_channels
        self.input_dimensions = input_dimensions    # (num_rows, num_columns)
        self.num_actions = num_actions
        self.conv_layer_sizes = conv_layer_sizes
        self.linear_sizes = linear_sizes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.lr = lr
        
        self.conv_layers = nn.ModuleList([])
        self.bn_layers = nn.ModuleList([])
        self.linear_layers = nn.ModuleList([])
        
        assert len(self.input_dimensions) == 2, 'Invalid input dimensions {}'.format(self.input_dimensions)
        
        if padding:
            # pad_text, l1_size = 'same', layer_1_size * 3 * 3
            # pad_text, l1_size = 'same', layer_2_size * 3 * 3
            pad_text, l1_size = 'same', self.conv_layer_sizes[-1] * self.input_dimensions[0] * self.input_dimensions[1]
        else:
            # pad_text, l1_size = 'valid', layer_1_size
            pad_text, l1_size = 'valid', self.conv_layer_sizes[-1] * (self.input_dimensions[0] - 2) * (self.input_dimensions[1] - 2)
                
        # self.conv1 = Conv2d(2, layer_1_size, 3, padding=pad_text)
        # self.bn1 = nn.BatchNorm2d(layer_1_size)
            
        # self.conv2 = Conv2d(layer_1_size, layer_2_size, 3, padding=pad_text)
        # self.bn2 = nn.BatchNorm2d(layer_2_size)
        
        # init conv, bn layers
        layer_input_channels = self.input_channels
        for conv_layer_size in self.conv_layer_sizes:
            conv = Conv2d(layer_input_channels, conv_layer_size, self.kernel_size, padding=pad_text)
            bn = nn.BatchNorm2d(conv_layer_size)
            self.conv_layers.append(conv)
            self.bn_layers.append(bn)
            
            layer_input_channels = conv_layer_size
        
        linear_input_size = l1_size
        for linear_size in self.linear_sizes:
            linear_layer = nn.Linear(linear_input_size, linear_size)
            self.linear_layers.append(linear_layer)
            
            linear_input_size = linear_size
        
        self.final_linear_layer = nn.Linear(linear_input_size, self.num_actions)
        
        # self.linear1 = nn.Linear(l1_size, linear_size)
        # self.linear2 = nn.Linear(linear_size, 9)

        self.activation = nn.LeakyReLU()
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.loss = nn.SmoothL1Loss()
        
        
    def forward(self, x):
        
        # c = self.conv1(x)
        # c = self.bn1(c)
        # c = self.activation(c)
        
        # c = self.conv2(c)
        # c = self.bn2(c)
        
        # c = c.flatten(start_dim=1)
        
        # c = self.activation(c)
        
        for conv, bn in zip(self.conv_layers, self.bn_layers):
            x = conv(x)
            x = bn(x)
            x = self.activation(x)
        
        x = x.flatten(start_dim=1)
        
        for lin in self.linear_layers:
            x = lin(x)
            x = self.activation(x)
        
        # l = self.linear1(c)
        # l = self.activation(l)
        # l = self.linear2(l)
        
        x = self.final_linear_layer(x)
        
        # l = torch.tanh(l)
        
        # TODO make this flexible, not just tanh
        x = torch.tanh(x)
        
        return x


# trial random game
random_agent = RandomAgent()

# model = TTTConvModel(32, 16)
# model = TTTConvModel(64, 32, 32)
# model = TTTConvModel(16, 16, 50)
model = TTTConvModel(2, (3, 3), 9, [16, 16], [50])
# model.to(dml)
model.eval()
gamma = 0
PER = True
dlagent = DLAgent(model, gamma=gamma, per=PER)


###

one_player_det = False

# playerX = TTTPlayer(1, random_agent)
playerX = TTTPlayer(1, dlagent)
playerO = TTTPlayer(-1, dlagent)
players = [playerX, playerO]

def play_series(players, num_games, one_player_det=False, title=None):
    
    if title is None:
        title = 'Ordinary'
    
    win_counter = [0 for p in players] + [0] # ties
    stashed_eps = [p.epsilon for p in players]
    
    for i in range(num_games):
        
        if one_player_det:
            # make a random player play determinatively, to reduce noise in training set
            det_player_index = random.randint(0, len(players)-1)
            players[det_player_index].epsilon = 0
        
        game = TicTacToe(players)
        game.play_game()

        if one_player_det:
            # reset epsilon to initial value
            players[det_player_index].epsilon = stashed_eps[det_player_index]
        
        if game.winner_index is None:
            win_counter[-1] += 1
        else:
            win_counter[game.winner_index] += 1
            
    names = ['X', 'O', 'Tie']
    plt.bar(names, win_counter)
    plt.title(title)
    plt.show()

def play_determinative(players, det_bools, num_games, title=None):
    
    assert len(players) == len(det_bools)
    
    stashed_eps = [p.epsilon for p in players]
    
    for p, p_det_bool in zip(players, det_bools):
        if p_det_bool:
            p.epsilon = 0
    
    play_series(players, num_games, title=title)

    for p, p_det_bool, eps in zip(players, det_bools, stashed_eps):
        if p_det_bool:
            p.epsilon = eps

start_time = time.time()
last_time = start_time
play_time, train_time = 0, 0

play_series(players, 1000, one_player_det=one_player_det)

play_time += time.time() - last_time
last_time = time.time()

for i in range(500):
    
    print()
    print(i)
    
    print('Training...')
    playerO.agent.train_n_batches(100)
    plt.plot(moving_average(playerO.agent.losses))
    plt.show()
    train_time += time.time() - last_time
    last_time = time.time()
    
    print('Playing...')
    play_series(players, 100)
    play_time += time.time() - last_time
    last_time = time.time()
    
    print('History length: {}'.format(playerX.agent.history.length))
    
    if i % 10 == 9:
        n_tests = 50
        print('Testing...')
        # todo should we not store history during this part?
        play_determinative(players, [True, False], n_tests, title='DetX')
        play_determinative(players, [False, True], n_tests, title='DetO')
        # play_determinative(players, [True, True], n_tests, title='DetBoth')

end_time = time.time()
tdiff = end_time - start_time

plt.plot(moving_average(playerO.agent.losses))

print('Time taken: {:.2f} s'.format(tdiff))
print('Play time: {:.2f} s'.format(play_time))
print('Train time: {:.2f} s'.format(train_time))

# print()
# profiler.display()
