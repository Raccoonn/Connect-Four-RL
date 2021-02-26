

import numpy as np



class ConnectFour:
    """
    Connect Four environment
        - 2 players
        - 6x7 board
        - Win if player connects 4 pieces in row, col, diag
    
    """

    def __init__(self):
        """
        Initialize environment

            - Player 1 is yellow
            - Player 2 is red

            - yellow = 1, red = -1

        """
        self.color_1 = 'yellow'
        self.color_2 = 'red'

        self.board = np.zeros((6,7))
        self.player = 1
        self.done = False






    def reset(self):
        """
        Reset game environment for new episode
        """
        # Reset game state
        self.board = np.zeros((6,7))
        self.player = 1
        self.done = False

        # No action taken prior to game start, action is just zeros
        action_vec = np.zeros(7)

        # np.append flattens the arrays together for input
        state = np.append(action_vec, self.board)

        return state




    def check_Full(self):
        """
        Check if the board is filled with pieces thus ending the game
        """
        if 0 in self.board:
            return False
        else:
            return True


 


    def check_Win(self):
        """
        Scan rows, columns, and diagonals for a connected sequence of matching pieces
            
            - Return True and player id if a win is found
            - Return False and None if there is no winner
        """
        winner = None
        # Check rows and columns
        for i in range(6):
            row = self.board[i, :]
            col = self.board[:, i]
        
            # Check by calculating sum of 4 element sections of rows and columns
            for idx in range(3):
                row_val = sum(row[idx : idx+4])
                col_val = sum(col[idx : idx+4])

                if row_val == 4 or col_val == 4:
                    self.done = True
                    winner = 1
                    return winner

                elif row_val == -4 or col_val == -4:
                    self.done = True
                    winner = 2
                    return winner


        # Check diagonals
        for off in range(-3, 4):
            left = np.diagonal(self.board, offset=off, axis1=1, axis2=0)
            right = np.diagonal(np.fliplr(self.board), offset=off, axis1=1, axis2=0)

            for idx in range(0, len(left)-3):
                left_val = sum(left[idx : idx+4])
                right_val = sum(right[idx : idx+4])

                if left_val == 4 or right_val == 4:
                    self.done = True
                    winner = 1
                    return winner

                elif left_val == -4 or right_val == -4:
                    self.done = True
                    winner = 2
                    return winner
                        

        # Return None if there is no winner      
        return winner





    def update_Board(self, action):
        """
        Given an action add the piece to the board, properly accounting for
        player turn and stacking in columns.

            - return True when a valid move has been made
            - return False when an invalid move is attempted
        """
        col = self.board[:, action]

        if self.player == 1:
            piece = 1
        else:
            piece = -1
        
        for i in range(-1, -7, -1):
            if col[i] == 0:
                col[i] = piece
                return True

        # If the selected move if invalid return false
        return False
            



    def update_Players(self):
        """
        Method to update players, useful to store state after action and before next turn.
        """
        if self.player == 1:
            self.player = 2
        else:
            self.player = 1




    def Step(self, action):
        """
        Take a Step:
            - Current player takes action
            - Return false if action invalid, player chooses again
            - Check board for win or draw
            - Update done if game is finished else update current player for next Step

        return format: True/False for move validity, state, reward, done, winner (1/2 if draw)


        """
        p1_rwd = 0
        p2_rwd = 0

        winner = None

        if self.update_Board(action) == True:
            valid = True
            winner = self.check_Win()

            # If board is full and no winner is found end game and return None for winner
            if self.check_Full() == True:
                self.done = True


            if self.done == True:
                if winner == 1:
                    p1_rwd += 1
                    p2_rwd -= 1
                elif winner == 2:
                    p1_rwd -= 1
                    p2_rwd += 1
                elif winner == None:
                    p1_rwd += 0.5
                    p2_rwd += 0.5

        # Catch for invalid move, prompting Player to choose another move
        # returns negative reward
        else:
            valid = False

            if self.player == 1:
                p1_rwd -= 1
            elif self.player == 2:
                p2_rwd -= 1

        
        # combine one hot + game board as the new state
        action_vec = np.zeros(7)
        action_vec[action] = 1

        state_ = np.append(action_vec, self.board)


        return valid, state_, p1_rwd, p2_rwd, winner


    











