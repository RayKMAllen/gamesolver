# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 15:19:13 2025

@author: raymo
"""

import numpy as np
import torch

from utils import idx_to_rowcol, rowcol_to_idx, onehot_mask


class Game:
    
    def __init__(self, players):
        self.players = players
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
        # overwriteable
        
        return 0
    
    def next_player(self):
        # overwriteable
        
        if self.current_player_index is None:
            self.current_player_index = self.first_player_index()
        else:
            self.current_player_index = (self.current_player_index + 1) % len(self.players)
                    
        self.current_player = self.players[self.current_player_index]



class Player:
    
    def __init__(self, player_id, game, agent=None, epsilon=0.1):
        self.player_id = player_id
        self.game = game
        self.epsilon = epsilon
        self.agent = agent
    
    def attach_game(self, game):
        self.game = game
    
    def turn(self):
        pass

class Board:
    
    def __init__(self):
        pass
        
    def show_board(self):
        
        print()
        for row in self.board:
            row_str = ' '.join([self.repr_map[c] for c in row])
            print(row_str)
        print()
        
class TTTBoard(Board):
    
    def __init__(self):
        super().__init__()
        self.board = np.array(
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
    
    # def show_board(self):
        
    #     print()
    #     for row in self.board:
    #         row_str = ' '.join([self.repr_map[c] for c in row])
    #         print(row_str)
    #     print()
    
    def is_valid_move(self, row, col):
        
        current = self.board[row][col]
        return current == 0        
    
    def place_piece(self, row, col, player_id):
        
        assert player_id in [-1, 1], 'Invalid player id {}'.format(player_id)
        
        assert self.is_valid_move(row, col), 'Index {}/{} of board is not 0, == {}'.format(row, col, self.board[row][col])
        
        self.board[row][col] = player_id

class TicTacToe(Game):
    
    def __init__(self, players, show=False):
        super().__init__(players)
        self.board = TTTBoard()
        # self.players = players
        self.show = show
        # self.current_player_index = None
        # self.current_player = None
        # self.winner = None
        # self.winner_index = None
        # self.rewards = None
        
        # self.states = {}
        # self.possible_actions = {}
        # self.actions = {}
                
        # for player in self.players:
        #     player.attach_game(self)
        #     self.states[player] = []
        #     self.possible_actions[player] = []
        #     self.actions[player] = []
        
    # def first_player_index(self):
        
    #     return 0
    
    # def next_player(self):
        
    #     if self.current_player_index is None:
    #         self.current_player_index = self.first_player_index()
    #     else:
    #         self.current_player_index = (self.current_player_index + 1) % len(self.players)
                    
    #     self.current_player = self.players[self.current_player_index]
    
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
        
        return state

class TTTPlayer(Player):
    
    def __init__(self, player_id, agent, epsilon=0.1):
        self.player_id = player_id
        self.game = None
        self.agent = agent
        self.epsilon = epsilon
        self.record = self.agent.history is not None   # bool, use history or not
        super().__init__(player_id, self.game, self.agent)

    def turn(self):
        
        assert self.game is not None, 'No game attached to player {}'.format(self.player_id)
        # profiler.log(3)
        
        encoded_state = self.game.encode_state(self)
        # profiler.log(4)
        if self.record:
            # record state
            self.game.states[self].append(encoded_state)
        
        possible_moves = self.game.possible_moves(self)
        # profiler.log(5)
        if self.record:
            # record possible actions
            self.game.possible_actions[self].append(onehot_mask(possible_moves, 9))
            # profiler.log(6)

        action = self.agent.choose(self.game, self, possible_moves, encoded_state=encoded_state)
        # profiler.log(7)
        if self.record:
            # record action
            self.game.actions[self].append(action)
        
        row, col = idx_to_rowcol(action)
        # profiler.log(8)
        self.game.board.place_piece(row, col, self.player_id)
        # profiler.log(9)
    
    def on_game_end(self, final_reward):
        
        if self.record:
            
            states = self.game.states[self]
            possible_actions = self.game.possible_actions[self]
            actions = self.game.actions[self]

            next_states = states[1:] + [states[-1]]   # last entry should not matter (fr_mix == 1)
            next_state_possible_actions = possible_actions[1:] + [possible_actions[-1]]
            
            assert len(actions) == len(next_states) == len(states)
            
            # mix final reward with stage-by-stage
            fr_mix = np.ones_like(actions, dtype=np.float32)
            for i in range(1, len(fr_mix)):
                fr_mix[i] = fr_mix[i-1]*self.agent.gamma
            # reverse
            fr_mix = fr_mix[::-1]    # 0-1
           
            for s, a, ns, nspa, fr_m in zip(states, actions, next_states, next_state_possible_actions, fr_mix):
                self.agent.history.update(s, a, ns, nspa, fr_m, final_reward)


class CheckersBoard(Board):
    
    def __init__(self):
        super().__init__()
        # initialize board
        self.board = np.zeros([8,8], dtype=int)
        for i in range(8):
            if i in [3,4]:
                continue
            init_player = -1 if i < 3 else 1
            for j in range(8):
                if (i+j)%2 == 1:
                    self.board[i][j] = init_player
                    
        self.repr_map = {
            0: '.',
            1: 'B',
            -1: 'W',
            2: 'K',
            -2: 'Q',    # white king
            }
        
    def jumped_square(self, fr, fc, tr, tc):
        
        row_diff, col_diff = tr - fr, tc - fc
        if row_diff not in [2,-2]:
            return None
        if col_diff not in [2,-2]:
            return None
        
        br, bc = (fr+tr)//2, (fc+tc)//2
        
        return [br, bc]
    
    def basic_move_check(self, player_id, fr, fc, tr, tc):
        
        if min(fr, fc, tr, tc) < 0:
            return False
        if max(fr, fc, tr, tc) > 7:
            return False

        current = self.board[fr][fc]
        if current not in [player_id, 2*player_id]:    # 2x for king
            return False
        if self.board[tr][tc] != 0:
            return False
        
        row_diff, col_diff = tr - fr, tc - fc
        # must be diagonal
        if col_diff not in [row_diff, -row_diff]:
            return False
        
        return True

    def is_valid_walk(self, player_id, fr, fc, tr, tc):
        # from row, from col, to row, to col
        
        if not self.basic_move_check(player_id, fr, fc, tr, tc):
            return False
        
        row_diff, col_diff = tr - fr, tc - fc
        
        # space vacant, one row ahead
        if row_diff == - player_id:
            # col check already done
            return True
        
        # king-backwards
        if row_diff == player_id and self.board[fr][fc] == 2 * player_id:
            return True
        
        return False
    
    def is_valid_capture(self, player_id, fr, fc, tr, tc):

        if not self.basic_move_check(player_id, fr, fc, tr, tc):
            return False
        
        row_diff, col_diff = tr - fr, tc - fc
        
        # single capture
        # regular or king-backwards
        if row_diff == -2 * player_id or (row_diff == 2 * player_id and self.board[fr][fc] == 2 * player_id):
            # get space between, check for opp player piece
            jumped_square = self.jumped_square(fr, fc, tr, tc)
            if jumped_square is not None:
                [br, bc] = jumped_square
                # if self.board[br][bc] == - player_id:
                if self.board[br][bc] in [-player_id, -2*player_id]:
                    return True
        
        return False
    
    def check_make_king(self, player_id, piece_id, tr, tc):

        if piece_id in [1, -1]:
            # make king check
            if player_id == 1 and tr == 0:
                # black king
                print('Black king!')
                self.board[tr][tc] = 2*player_id
            elif player_id == -1 and tr == 7:
                # white king
                print('White king!')
                self.board[tr][tc] = 2*player_id        
    
    def do_walk(self, player_id, fr, fc, tr, tc):
        
        assert self.is_valid_walk(player_id, fr, fc, tr, tc), 'Invalid walk: player {}, from {}/{}, to {}/{}'.format(player_id, fr, fc, tr, tc)
        piece_id = self.board[fr][fc]    # piece or king
        
        self.board[fr][fc] = 0
        self.board[tr][tc] = piece_id
        
        self.check_make_king(player_id, piece_id, tr, tc)
    
    def do_capture(self, player_id, fr, fc, tr, tc):
        
        assert self.is_valid_capture(player_id, fr, fc, tr, tc), 'Invalid capture: player {}, from {}/{}, to {}/{}'.format(player_id, fr, fc, tr, tc)
        piece_id = self.board[fr][fc]    # piece or king
        
        self.board[fr][fc] = 0
        self.board[tr][tc] = piece_id        
        jumped_square = self.jumped_square(fr, fc, tr, tc)
        if jumped_square is None:
            raise Exception('Invalid capture: player {} jumped square is None ({}/{} to {}/{})'.format(player_id, fr, fc, tr, tc))
        [br, bc] = jumped_square
        self.board[br][bc] = 0

        self.check_make_king(player_id, piece_id, tr, tc)
        
class Checkers(Game):
    
    def __init__(self, players, show=False):
        super().__init__(players)
        self.board = CheckersBoard()
        self.show = show
        self.num_turns = 0
    
    def possible_moves(self, player):
    
        player_id = player.player_id

        possible_captures = []
        # TODO make this more efficient
        for fr in range(8):
            for fc in range(8):
                for tr in [fr-2, fr+2]:
                    for tc in [fc-2, fc+2]:
                        if self.board.is_valid_capture(player_id, fr, fc, tr, tc):
                            possible_captures.append((player_id, fr, fc, tr, tc))
        
        # if any capture is possible, must do one of them
        if len(possible_captures) > 0:
            return possible_captures
        
        # otherwise look at walks
        possible_walks = []
        for fr in range(8):
            for fc in range(8):
                for tr in [fr-1, fr+1]:
                    for tc in [fc-1, fc+1]:
                        if self.board.is_valid_walk(player_id, fr, fc, tr, tc):
                            possible_walks.append((player_id, fr, fc, tr, tc))
        
        return possible_walks
    
    def possible_captures_from(self, player, fr, fc):
        # for multicaptures
        
        player_id = player.player_id

        possible_captures = []
        for tr in [fr-2, fr+2]:
            for tc in [fc-2, fc+2]:
                if self.board.is_valid_capture(player_id, fr, fc, tr, tc):
                    possible_captures.append((player_id, fr, fc, tr, tc))
        
        return possible_captures

    
    def is_game_over(self):
        
        for i, player in enumerate(self.players):
            
            player_id = player.player_id
            
            if (self.board.board == player_id).sum() == 0:
                # player loses
                opponent = self.players[(i+1)%2]
                self.on_game_end(winner=opponent)
                return True
        
        return False
    
    def on_game_end(self, winner):
        
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
            
  
    def play_game(self):
        
        if self.show:
            self.board.show_board()

        while not self.is_game_over():
                        
            self.next_player()
            early_exit = self.current_player.turn()
            self.num_turns += 1

            if self.show:
                self.board.show_board()
            
            if early_exit:
                break

    def encode_state(self, player):
        
        player_id = player.player_id
        
        # TODO add kings
        me_mask = torch.tensor(self.board.board == player_id, dtype=torch.float32)
        me_king_mask = torch.tensor(self.board.board == 2*player_id, dtype=torch.float32)
        them_mask = torch.tensor(self.board.board == -player_id, dtype=torch.float32)
        them_king_mask = torch.tensor(self.board.board == -2*player_id, dtype=torch.float32)
        
        # what should order be?
        state = torch.stack([me_mask, me_king_mask, them_mask, them_king_mask])
        
        return state

class CheckersPlayer(Player):
    
    def __init__(self, player_id, agent, epsilon=0.1):
        self.player_id = player_id
        self.game = None
        self.agent = agent
        self.epsilon = epsilon
        self.record = self.agent.history is not None   # bool, use history or not
        super().__init__(player_id, self.game, self.agent)

    def turn(self):
        
        assert self.game is not None, 'No game attached to player {}'.format(self.player_id)
        
        encoded_state = self.game.encode_state(self)
        if self.record:
            # record state
            self.game.states[self].append(encoded_state)
        
        possible_moves = self.game.possible_moves(self)
        # check for game end
        if len(possible_moves) == 0:
            # player loses
            opponent = self.game.players[(self.game.current_player_index+1)%2]
            self.game.on_game_end(winner=opponent)
            # early exit
            return True

        if self.record:
            # record possible actions
            # TODO think about how to encode
            self.game.possible_actions[self].append(possible_moves)

        print('Possible moves:')
        print(possible_moves)
        action = self.agent.choose(self.game, self, possible_moves, encoded_state=encoded_state)
        if self.record:
            # record action
            self.game.actions[self].append(action)
        
        # row, col = idx_to_rowcol(action)
        # self.game.board.place_piece(row, col, self.player_id)
        (pid, fr, fc, tr, tc) = action
        print('Action: ', action)
        assert pid == self.player_id, 'Invalid action player id'
        
        capture = (tr - fr) in [-2, 2]
        if capture:
            self.game.board.do_capture(pid, fr, fc, tr, tc)
            
            # TODO multi-captures
            possible_multicaptures = self.game.possible_captures_from(self, tr, tc)
            while len(possible_multicaptures) > 0:
                
                print('Multicapturing!')

                if self.game.show:
                    self.game.board.show_board()

                encoded_state = self.game.encode_state(self)
                action = self.agent.choose(self.game, self, possible_multicaptures, encoded_state=encoded_state)
                if self.record:
                    # record state etc
                    self.game.states[self].append(encoded_state)
                    self.game.possible_actions[self].append(possible_multicaptures)
                    self.game.actions[self].append(action)

                (pid, fr, fc, tr, tc) = action
                self.game.board.do_capture(pid, fr, fc, tr, tc)

                possible_multicaptures = self.game.possible_captures_from(self, tr, tc)                
            
        else:
            self.game.board.do_walk(pid, fr, fc, tr, tc)
        
        # early exit bool
        return False
    
    def on_game_end(self, final_reward):
        
        if self.record:
            
            states = self.game.states[self]
            possible_actions = self.game.possible_actions[self]
            actions = self.game.actions[self]

            next_states = states[1:] + [states[-1]]   # last entry should not matter (fr_mix == 1)
            next_state_possible_actions = possible_actions[1:] + [possible_actions[-1]]
            
            assert len(actions) == len(next_states) == len(states)
            
            # mix final reward with stage-by-stage
            fr_mix = np.ones_like(actions, dtype=np.float32)
            for i in range(1, len(fr_mix)):
                fr_mix[i] = fr_mix[i-1]*self.agent.gamma
            # reverse
            fr_mix = fr_mix[::-1]    # 0-1
           
            for s, a, ns, nspa, fr_m in zip(states, actions, next_states, next_state_possible_actions, fr_mix):
                self.agent.history.update(s, a, ns, nspa, fr_m, final_reward)


board = CheckersBoard()
# board.show_board()
# board.place_piece(3, 4, 1)
# board.show_board()

# board.board[3][2] = 1
# board.board[4][3] = 1
# board.board[3][2] = -1
# board.board[4][3] = -1
# board.show_board()

# # cumbersome move check
# for player_id in [1, -1]:
#     for fr in range(8):
#         for fc in range(8):
#             for tr in range(8):
#                 for tc in range(8):
#                     if board.is_valid_move(player_id, fr, fc, tr, tc):
#                         print(player_id, fr, fc, tr, tc)
#                         board.move_piece(player_id, fr, fc, tr, tc)
#                         board.show_board()


from agent import RandomAgent

rand_agent = RandomAgent()

p1 = CheckersPlayer(1, rand_agent)
p2 = CheckersPlayer(-1, rand_agent)
players = [p1, p2]

game = Checkers(players, show=True)
game.play_game()


