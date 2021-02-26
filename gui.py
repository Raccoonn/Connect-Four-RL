
from tkinter import *
from tkinter import font

import numpy as np
from dqn import Agent




"""
    Changes:
        - Need to modify function for adding pieces

        - Easier method for tracking player moves, this needs to be added to board
          state to make the 7*7 input with one hot vector for previous move

"""



def setup_Agent(filename, epsilon):
    """
    Function to initialize the DQN agent
    """
    input_dims = 6*7
    action_space = tuple(range(7))
    n_actions = 7

    h1_dims = 512
    h2_dims = 256


    agent = Agent(lr=0.001, gamma=0.95, epsilon=epsilon, epsilon_dec=0.995, epsilon_min=0.01,
                  input_shape=input_dims, h1_dims=h1_dims, h2_dims=h2_dims, action_space=action_space,
                  training_epochs=1, fname=filename
                  )

    return agent





class Info(Frame):
    """
    Initialize information frame, used for displaying player turns and winner.

    NEEDS:
        - Display for Agent choice given the game state
    
    """
    def __init__(self, master=None):
        Frame.__init__(self)
        self.configure(width=500, height=100)
        police = font.Font(self, size=20, family='Arial')
        self.t = Label(self, text='Connect Four', font=police)
        self.t.grid(sticky=NSEW, pady=20)





class Piece(object):
    """
    Create object to represent each piece location on the game board
    """
    def __init__(self, x, y, canv, color='white', bg='red'):
        self.canv = canv
        self.x = x
        self.y = y
        self.color = color

        self.tour = 1

        self.r = 30
        self.piece = self.canv.create_oval(self.x+10, self.y+10, self.x+61, self.y+61,
                                          fill=color, outline='blue')

    
    def change_Color(self, color):
        """
        Method to change color of pieces for each player and resets
        """
        self.canv.itemconfigure(self.piece, fill=color)
        self.color = color






class Terrain(Canvas):
    """
    Canvas object to represent game board
    """
    def __init__(self, master=None):
        Canvas.__init__(self)
        self.configure(width=500, height=400, bg='blue')

        self.player = 1
        self.color = 'yellow'
        self.board = []
        self.done = False

        for i in range(0, 340, 400//6):
            list_row = []
            for j in range(0, 440, 500//7):
                list_row.append(Piece(j, i, self))

            self.board.append(list_row)
        

        # Bind left click to call detect column to fill board
        # Passes the mouse button event to the detection method
        self.bind('<Button-1>', self.Step)


        self.agent_1 = setup_Agent('p1.h5', epsilon=0)
        self.agent_2 = setup_Agent('p2.h5', epsilon=0)

        self.agent_1.load_model('p1.h5')
        self.agent_2.load_model('p2.h5')

        state = self.get_Board().flatten()

        action, actions = self.agent_1.choose_action(state)


        info.t.config(text='P1 Predicts:' + str(actions) + '\nArgMax: ' + str(action+1),
                      font=('Arial', 12))




    def get_Board(self):
        """
        Returns the board as a 2D array
            
            - yellow = 1
            - red = -1
            - empty = 0
        """
        board_array = np.zeros((6,7))
        for i in range(6):
            for j in range(7):
                if self.board[i][j].color == 'yellow':
                    board_array[i,j] = 1

                elif self.board[i][j].color == 'red':
                    board_array[i,j] = -1

        return board_array





    def check_Win(self, board_array):
        """
        Scan rows, columns, and diagonals for a connected sequence of matching pieces
        
        Check by calculating sum of 4 element sections of rows, columns, and diagonals
            
            - Updates self.done and returns winning player number or None if no winner

        """
        winner = None
        # Check rows and columns
        for i in range(6):
            row = board_array[i, :]
            col = board_array[:, i]
        
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
            left = np.diagonal(board_array, offset=off, axis1=1, axis2=0)
            right = np.diagonal(np.fliplr(board_array), offset=off, axis1=1, axis2=0)

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
                        

        #Return False and None if there is no winner      
        return winner




    def check_Full(self, board_array):
        """
        Check if the board is filled with pieces thus ending the game
        """
        if 0 in board_array:
            return False
        else:
            return True




    def Step(self, event):
        """
        Given a column selection, place the piece in the available spot

        Check if there is a winner and check if board is full resulting in a draw
        """
        if not self.done:
            col = event.x//71
            row = 0

            # Loop to find available spot for piece
            while row < len(self.board):
                if self.board[0][col].color == 'red' or self.board[0][0].color == 'yellow':
                    break

                if self.board[row][col].color == 'red' or self.board[row][col].color == 'yellow':
                    self.board[row-1][col].change_Color(self.color)
                    break
                
                elif row == len(self.board)-1:
                    self.board[row][col].change_Color(self.color)
                    break


                if self.board[row][col].color != 'red' and self.board[row][col].color != 'yellow':
                    row += 1



            # Convert game board to 2D array and check for winners
            # If game is not over change turn and continue

            board_array = self.get_Board()
            winner = self.check_Win(board_array)


            state = board_array.flatten()


            if winner == 1:
                info.t.config(text='Player 1 Wins!')
            
            elif winner == 2:
                info.t.config(text='Player 2 Wins!')

            elif winner == None and self.check_Full(board_array) == True:
                info.t.config(text='Game is a Draw')


            else:
                # Change player and update info frame after each move
                if self.player == 1:
                    self.player = 2
                    action, actions = self.agent_2.choose_action(state)
                    info.t.config(text='P2 predicts:' + str(actions) + '\nArgMax: ' + str(action+1),
                      font=('Arial', 12))
                    self.color = 'red'

                elif self.player == 2:
                    self.player = 1
                    action, actions = self.agent_1.choose_action(state)
                    info.t.config(text='P1 predicts:' + str(actions) + '\nArgMax: ' + str(action+1),
                      font=('Arial', 12))
                    self.color = 'yellow'









if __name__ == '__main__':

    
    root = Tk()
    root.geometry("500x550")
    root.title("Puissance 4 - V 1.0 -- Romain VAUSE")   # Mad credit to this homie

    info = Info(root)
    info.grid(row=0, column=0)


    t = Terrain(root)
    t.grid(row=1, column=0)

    def rein():
        global info
        info.t.config(text="")
        
        info = Info(root)
        info.grid(row=0, column=0)

        t = Terrain(root)
        t.grid(row=1, column=0)

    Button(root, text="Reset", command=rein).grid(row=2, column=0, pady=30)

    root.mainloop()

