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
from sumtree import SumTree

nest_asyncio.apply()

# dml = torch_directml.device()    

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

profiler = Profiler()

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
        


class History:

    def __init__(self, max_len=100000, per=False, eps=1e-2, alpha=0.1, beta=0.1):

        # self.states = deque()
        # self.actions = deque()
        # self.next_states = deque()
        # self.next_state_possible_actions = deque()
        # self.fr_mix = deque()
        # self.final_rewards = deque()
        self.states = [None] * max_len
        self.actions = [None] * max_len
        self.next_states = [None] * max_len
        self.next_state_possible_actions = [None] * max_len
        self.fr_mix = [None] * max_len
        self.final_rewards = [None] * max_len
        
        
        self.max_len = max_len
        self.per = per    # Prioritized Experience Replay
        self.length = 0
        self.counter = 0

        if self.per:
            
            self.tree = SumTree(size=max_len)
            
            # PER params
            self.eps = eps  # minimal priority, prevents zero probabilities
            self.alpha = alpha  # determines how much prioritization is used, α = 0 corresponding to the uniform case
            self.beta = beta  # determines the amount of importance-sampling correction, b = 1 fully compensate for the non-uniform probabilities
            self.max_priority = eps  # priority for new samples, init as eps

    def update(self, state, action, next_state, nspa, fr_mix, final_reward):
        
        # self.states.append(state)
        # self.actions.append(action)
        # self.next_states.append(next_state)
        # self.next_state_possible_actions.append(nspa)
        # self.fr_mix.append(fr_mix)
        # self.final_rewards.append(final_reward)
        # self.length += 1
        
        if self.per:
            # store transition index with maximum priority in sum tree
            self.tree.add(self.max_priority, self.counter)

        self.states[self.counter] = state
        self.actions[self.counter] = action
        self.next_states[self.counter] = next_state
        self.next_state_possible_actions[self.counter] = nspa
        self.fr_mix[self.counter] = fr_mix
        self.final_rewards[self.counter] = final_reward
        
        # if len(self.actions) > self.max_len:

        #     for dat in [self.states, self.actions, self.next_states, self.next_state_possible_actions, self.fr_mix, self.final_rewards]:
        #         dat.popleft()
        #         assert len(dat) <= self.max_len
        #     self.length -= 1
            
        # assert self.length == len(self.actions), 'Invalid length for History'

        self.counter = (self.counter + 1) % self.max_len
        self.length = min(self.max_len, self.length + 1)

    def data_from_indexes(self, indexes):
        
        batch_states = [self.states[i] for i in indexes]
        batch_actions = [self.actions[i] for i in indexes]
        batch_next_states = [self.next_states[i] for i in indexes]
        batch_nspas = [self.next_state_possible_actions[i] for i in indexes]
        batch_fr_mix = [self.fr_mix[i] for i in indexes]
        batch_final_rewards = [self.final_rewards[i] for i in indexes]
        
        batch_states = torch.stack(batch_states)
        batch_actions = torch.tensor(batch_actions, dtype=int)
        batch_next_states = torch.stack(batch_next_states)
        batch_nspas = torch.tensor(batch_nspas)
        batch_fr_mix = torch.tensor(batch_fr_mix, dtype=torch.float32)
        batch_final_rewards = torch.tensor(batch_final_rewards, dtype=torch.float32)
        
        return batch_states, batch_actions, batch_next_states, batch_nspas, batch_fr_mix, batch_final_rewards

    def sample_uniform(self, batch_size):
        bs = batch_size if len(self.actions) >= batch_size else len(self.actions)
        
        # choose randomly without deletion
        sample_idxs = random.sample(range(self.length), bs)
        
        # batch_states = [self.states[i] for i in batch_idxs]
        # batch_actions = [self.actions[i] for i in batch_idxs]
        # batch_next_states = [self.next_states[i] for i in batch_idxs]
        # batch_nspas = [self.next_state_possible_actions[i] for i in batch_idxs]
        # batch_fr_mix = [self.fr_mix[i] for i in batch_idxs]
        # batch_final_rewards = [self.final_rewards[i] for i in batch_idxs]
        
        # batch_states = torch.stack(batch_states)
        # batch_actions = torch.tensor(batch_actions, dtype=int)
        # batch_next_states = torch.stack(batch_next_states)
        # batch_nspas = torch.tensor(batch_nspas)
        # batch_fr_mix = torch.tensor(batch_fr_mix, dtype=torch.float32)
        # batch_final_rewards = torch.tensor(batch_final_rewards, dtype=torch.float32)
        
        batch_states, batch_actions, batch_next_states, batch_nspas, batch_fr_mix, batch_final_rewards = self.data_from_indexes(sample_idxs)
        
        return batch_states, batch_actions, batch_next_states, batch_nspas, batch_fr_mix, batch_final_rewards
    
    def sample_per(self, batch_size):
        bs = batch_size if len(self.actions) >= batch_size else len(self.actions)
        
        sample_idxs, tree_idxs = [], []
        priorities = torch.empty(bs, 1, dtype=torch.float)

        # To sample a minibatch of size k, the range [0, p_total] is divided equally into k ranges.
        # Next, a value is uniformly sampled from each range. Finally the transitions that correspond
        # to each of these sampled values are retrieved from the tree. (Appendix B.2.1, Proportional prioritization)
        segment = self.tree.total / batch_size
        for i in range(batch_size):
            a, b = segment * i, segment * (i + 1)

            cumsum = random.uniform(a, b)
            # sample_idx is a sample index in buffer, needed further to sample actual transitions
            # tree_idx is a index of a sample in the tree, needed further to update priorities
            tree_idx, priority, sample_idx = self.tree.get(cumsum)

            priorities[i] = priority
            tree_idxs.append(tree_idx)
            sample_idxs.append(sample_idx)
        
        probs = priorities / self.tree.total
        
        weights = (self.length * probs) ** -self.beta
        weights = weights / weights.max()
        
        batch_states, batch_actions, batch_next_states, batch_nspas, batch_fr_mix, batch_final_rewards = self.data_from_indexes(sample_idxs)
        
        return batch_states, batch_actions, batch_next_states, batch_nspas, batch_fr_mix, batch_final_rewards, weights, tree_idxs
    
    def yield_batch(self, batch_size):
        
        if not self.per:
            
            return self.sample_uniform(batch_size)
        
        else:
            
            return self.sample_per(batch_size)

    def update_priorities(self, data_idxs, priorities):
        if isinstance(priorities, torch.Tensor):
            priorities = priorities.detach().cpu().numpy()

        for data_idx, priority in zip(data_idxs, priorities):
            # The first variant we consider is the direct, proportional prioritization where p_i = |δ_i| + eps,
            # where eps is a small positive constant that prevents the edge-case of transitions not being
            # revisited once their error is zero. (Section 3.3)
            priority = (priority + self.eps) ** self.alpha

            self.tree.update(data_idx, priority)
            self.max_priority = max(self.max_priority, priority)        

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

    def __init__(self, layer_1_size, layer_2_size, linear_size, lr=1e-3, padding=True):
        super().__init__()
        self.layer_1_size = layer_1_size
        self.layer_2_size = layer_2_size
        self.linear_size = linear_size
        self.lr = lr
        
        if padding:
            # pad_text, l1_size = 'same', layer_1_size * 3 * 3
            pad_text, l1_size = 'same', layer_2_size * 3 * 3
        else:
            # pad_text, l1_size = 'valid', layer_1_size
            pad_text, l1_size = 'valid', layer_2_size
            
        self.conv1 = Conv2d(2, layer_1_size, 3, padding=pad_text)
        self.bn1 = nn.BatchNorm2d(layer_1_size)
            
        self.conv2 = Conv2d(layer_1_size, layer_2_size, 3, padding=pad_text)
        self.bn2 = nn.BatchNorm2d(layer_2_size)
        
        self.linear1 = nn.Linear(l1_size, linear_size)
        self.linear2 = nn.Linear(linear_size, 9)

        self.activation = nn.LeakyReLU()
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.loss = nn.SmoothL1Loss()
        
        
    def forward(self, x):
        
        c = self.conv1(x)
        c = self.bn1(c)
        c = self.activation(c)
        
        c = self.conv2(c)
        c = self.bn2(c)
        
        c = c.flatten(start_dim=1)
        
        c = self.activation(c)
        
        l = self.linear1(c)
        l = self.activation(l)
        l = self.linear2(l)
        
        l = torch.tanh(l)
        
        return l    


# trial random game
random_agent = TTTRandomAgent()

# model = TTTConvModel(32, 16)
# model = TTTConvModel(64, 32, 32)
model = TTTConvModel(16, 16, 50)
# model.to(dml)
model.eval()
gamma = 0.65
PER = True
dlagent = DLAgent(model, gamma=gamma, per=PER)

# playerX = TTTPlayer(1, random_agent)
playerX = TTTPlayer(1, dlagent)
playerO = TTTPlayer(-1, dlagent)
players = [playerX, playerO]

def play_series(players, num_games, title=None):
    
    if title is None:
        title = 'Ordinary'
    
    win_counter = [0 for p in players] + [0] # ties

    for i in range(num_games):
        game = TicTacToe(players)
        game.play_game()
        
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

play_series(players, 1000)

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
    
    if i % 10 == 9:
        # n_tests = 100
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
