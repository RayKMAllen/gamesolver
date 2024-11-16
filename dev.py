# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 12:45:16 2024

@author: raymo
"""

import numpy as np

class Agent:
    
    def __init__(self):
        pass
    


class Game:
    
    def __init__(self):
        pass



class Player:
    
    def __init__(self, player_id, game, agent=None):
        self.player_id = player_id
        self.game = game
        self.agent = agent
    
    def turn(self):
        pass

def idx_to_rowcol(idx):
    
    row = idx // 3
    col = idx % 3
    
    return row, col

def rowcol_to_idx(row, col):
    
    return 3 * row + col

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
    
    def __init__(self, players):
        super().__init__()
        self.board = TTTBoard()
        self.players = players
        self.current_player_index = None
        self.current_player = None
        self.winner = None
        
        for player in self.players:
            player.attach_game(self)
        
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
                
                # row
                if (self.board.board[i] == player_id).all():
                    # print('Player {} wins'.format(player_id))
                    # self.winner = player
                    self.on_game_end(player)
                    return True
                
                # col
                if (self.board.board[:,i] == player_id).all():
                    # print('Player {} wins'.format(player_id))
                    # self.winner = player
                    self.on_game_end(player)
                    return True
                
                # diags
                if (self.board.board.diagonal() == player_id).all():
                    self.on_game_end(player)
                    return True
                if (self.board.board[::-1].diagonal() == player_id).all():
                    self.on_game_end(player)
                    return True
        
        # check for tie (all places filled)
        if len(self.possible_moves(self.current_player)) == 0:
            # print('Tie')
            self.on_game_end(None)
            return True
        
        return False
    
    def on_game_end(self, winner):
        
        self.board.show_board()
        
        if winner is None:
            print('Tie')
        
        else:
            print('Player {} wins'.format(winner.player_id))
            self.winner = winner
            

    
    def play_game(self):
        
        self.board.show_board()

        while not self.is_game_over():
                        
            self.next_player()
            self.current_player.turn()

            self.board.show_board()
        

class TTTPlayer(Player):
    
    def __init__(self, player_id, agent):
        self.player_id = player_id
        self.game = None
        self.agent = agent
        super().__init__(player_id, self.game, self.agent)
    
    def attach_game(self, game):
        self.game = game
    
    def turn(self):
        
        assert self.game is not None, 'No game attached to player {}'.format(self.player_id)
        
        possible_moves = self.game.possible_moves(self)
        action = self.agent.choose(possible_moves)
        
        row, col = idx_to_rowcol(action)
        self.game.board.place_piece(row, col, self.player_id)
            

# board = TTTBoard()
# board.show_board()
# board.place_piece(1, 2, 1)
# board.show_board()
# board.place_piece(1, 1, -1)
# board.show_board()

class TTTRandomAgent(Agent):
    
    def __init__(self):
        super().__init__()
    
    def choose(self, possible_moves):
        
        return np.random.choice(possible_moves)

# trial random game
random_agent = TTTRandomAgent()
playerX = TTTPlayer(1, random_agent)
playerO = TTTPlayer(-1, random_agent)
players = [playerX, playerO]

game = TicTacToe(players)
game.play_game()
