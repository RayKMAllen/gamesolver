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

from utils import Profiler
from history import History

nest_asyncio.apply()

class Agent:
    
    def __init__(self, gamma=0.95):
        self.gamma = gamma          # reward discount per step
        # self.epsilon = epsilon      # exploration control
        pass
    


class Game:
    
    def __init__(self):
        pass



class Player:
    
    def __init__(self, player_id, game, agent=None, epsilon=0.1):
        self.player_id = player_id
        self.game = game
        self.epsilon = epsilon
        self.agent = agent
    
    def turn(self):
        pass

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

class TTTBoard:
    
    def __init__(self):
        self.board = np.array(
            # [0] * 9
            [
            [0,0,0],
            [0,0,0],
            [0,0,0],
                ]
            )
        self.repr_map = {
            0: '.',
            1: 'X',
            -1: 'O',
            }
    
    def show_board(self):
        
        print()
        for row in self.board:
            row_str = ' '.join([self.repr_map[c] for c in row])
            print(row_str)
        print()
    
    def is_valid_move(self, row, col):
        
        current = self.board[row][col]
        return current == 0        
    
    def place_piece(self, row, col, player_id):
        
        assert player_id in [-1, 1], 'Invalid player id {}'.format(player_id)
        
        assert self.is_valid_move(row, col), 'Index {}/{} of board is not 0, == {}'.format(row, col, self.board[row][col])
        
        self.board[row][col] = player_id

class TicTacToe(Game):
    
    def __init__(self, players, show=False):
        super().__init__()
        self.board = TTTBoard()
        self.players = players
        self.show = show
        self.current_player_index = None
        self.current_player = None
        self.winner = None
        self.winner_index = None
        self.rewards = None
        
        self.states = {}
        self.possible_actions = {}
        self.actions = {}
                
        for player in self.players:
            player.attach_game(self)
            self.states[player] = []
            self.possible_actions[player] = []
            self.actions[player] = []
        
    def first_player_index(self):
        
        return 0
    
    def next_player(self):
        
        if self.current_player_index is None:
            self.current_player_index = self.first_player_index()
        else:
            self.current_player_index = (self.current_player_index + 1) % len(self.players)
                    
        self.current_player = self.players[self.current_player_index]
    
    def possible_moves(self, player):
        
        possible_moves = []
        for i in range(9):
            row, col = idx_to_rowcol(i)
            if self.board.is_valid_move(row, col):
                possible_moves.append(i)
        
        return possible_moves
    
    def is_game_over(self):
        
        for player in self.players:
            
            player_id = player.player_id
            
            for i in range(3):

                # profiler.log('20')
                
                # row
                if (self.board.board[i] == player_id).all():
                    # profiler.log('21')
                    self.on_game_end(player)
                    return True
                # profiler.log('21')
                
                # col
                if (self.board.board[:,i] == player_id).all():
                    # profiler.log('22')
                    self.on_game_end(player)
                    return True
                # profiler.log('22')
                
                # diags
                if (self.board.board.diagonal() == player_id).all():
                    # profiler.log('23')
                    self.on_game_end(player)
                    return True
                # profiler.log('23')
                if (self.board.board[::-1].diagonal() == player_id).all():
                    # profiler.log('24')
                    self.on_game_end(player)
                    return True
                # profiler.log('24')
        
        # check for tie (all places filled)
        if len(self.possible_moves(self.current_player)) == 0:
            self.on_game_end(None)
            return True
        
        return False
    
    def on_game_end(self, winner):

        # profiler.log('28')
        
        if self.show:
            self.board.show_board()
        
        if winner is None:
            if self.show:
                print('Tie')
            self.rewards = [0,0]
        
        else:
            if self.show:
                print('Player {} wins'.format(winner.player_id))
            self.winner = winner
            
            self.rewards = []
            for i, player in enumerate((self.players)):
                if player == winner:
                    reward = 1
                    self.winner_index = i
                else:
                    reward = -1
                
                self.rewards.append(reward)
                
                player.on_game_end(reward)

        # profiler.log('29')
            
  
    def play_game(self):
        
        # for player in self.players:
        #     player.clear()
        
        if self.show:
            self.board.show_board()
        
        # profiler.log(0)

        while not self.is_game_over():

            # profiler.log(1)
                        
            self.next_player()
            self.current_player.turn()

            # profiler.log(2)

            if self.show:
                self.board.show_board()

    def encode_state(self, player):
        
        player_id = player.player_id
        
        me_mask = torch.tensor(self.board.board == player_id, dtype=torch.float32)
        them_mask = torch.tensor(self.board.board == -player_id, dtype=torch.float32)
        state = torch.stack([me_mask, them_mask])
        
        # me_mask = np.array(self.board.board == player_id, dtype=np.float32)
        # them_mask = np.array(self.board.board == -player_id, dtype=np.float32)
        
        # state = np.stack([me_mask, them_mask])
        
        return state

class TTTPlayer(Player):
    
    def __init__(self, player_id, agent, epsilon=0.1):
        self.player_id = player_id
        self.game = None
        self.agent = agent
        self.epsilon = epsilon
        self.record = self.agent.history is not None   # bool, use history or not
        super().__init__(player_id, self.game, self.agent)
        # self.clear()
    
    def attach_game(self, game):
        self.game = game

    # def clear(self):
    #     # at init and game start
        
    #     self.states = []
    #     self.possible_actions = []
    #     self.actions = []
        
    def turn(self):
        
        assert self.game is not None, 'No game attached to player {}'.format(self.player_id)
        # profiler.log(3)
        
        encoded_state = self.game.encode_state(self)
        # profiler.log(4)
        if self.record:
            # record state
            # self.states.append(encoded_state)
            self.game.states[self].append(encoded_state)
        
        possible_moves = self.game.possible_moves(self)
        # profiler.log(5)
        if self.record:
            # record possible actions
            # self.possible_actions.append(onehot_mask(possible_moves, 9))
            self.game.possible_actions[self].append(onehot_mask(possible_moves, 9))
            # profiler.log(6)

        action = self.agent.choose(self.game, self, possible_moves, encoded_state=encoded_state)
        # profiler.log(7)
        if self.record:
            # record action
            # self.actions.append(action)
            self.game.actions[self].append(action)
        
        row, col = idx_to_rowcol(action)
        # profiler.log(8)
        self.game.board.place_piece(row, col, self.player_id)
        # profiler.log(9)
    
    def on_game_end(self, final_reward):
        
        if self.record:
            
            # states = self.states
            # next_states = self.states[1:] + [states[-1]]   # last entry should not matter (fr_mix == 1)
            # next_state_possible_actions = self.possible_actions[1:] + [self.possible_actions[-1]]
            
            states = self.game.states[self]
            possible_actions = self.game.possible_actions[self]
            actions = self.game.actions[self]

            next_states = states[1:] + [states[-1]]   # last entry should not matter (fr_mix == 1)
            next_state_possible_actions = possible_actions[1:] + [possible_actions[-1]]
            
            # assert len(self.actions) == len(next_states) == len(states)
            assert len(actions) == len(next_states) == len(states)
            
            # mix final reward with stage-by-stage
            # fr_mix = np.ones_like(self.actions, dtype=np.float32)
            fr_mix = np.ones_like(actions, dtype=np.float32)
            for i in range(1, len(fr_mix)):
                fr_mix[i] = fr_mix[i-1]*self.agent.gamma
            # reverse
            fr_mix = fr_mix[::-1]    # 0-1
           
            # for s, a, ns, nspa, fr_m in zip(states, self.actions, next_states, next_state_possible_actions, fr_mix):
            for s, a, ns, nspa, fr_m in zip(states, actions, next_states, next_state_possible_actions, fr_mix):
                self.agent.history.update(s, a, ns, nspa, fr_m, final_reward)
        

class TTTRandomAgent(Agent):
    
    def __init__(self):
        super().__init__()
        self.history = None
        
    def choose(self, game, player, possible_moves, encoded_state=None):
        
        move = np.random.choice(possible_moves)
        # profiler.log('rchoice')
        return move

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
random_agent = TTTRandomAgent()

# model = TTTConvModel(32, 16)
# model = TTTConvModel(64, 32, 32)
# model = TTTConvModel(16, 16, 50)
model = TTTConvModel(2, (3, 3), 9, [16, 16], [50])
# model.to(dml)
model.eval()
gamma = 0
PER = True
dlagent = DLAgent(model, gamma=gamma, per=PER)

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

for i in range(1000):
    
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
