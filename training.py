
from dqn import Agent
from environment import ConnectFour
from training_tools import *

import time



"""
    TRAINING PROGRESS:

        - Change activation layers to LeakyReLU

        - 2 Dense, 1 output, no output activation


"""









"""
Functions to setup various game playing agents
"""

def random_Agent(board):
    """
    Given a board state the random_Agent will return a random valid move
    
        NOTE: Random agent takes board, NOT input state
    """
    poss = []
    for j in range(7):
        if board[0,j] == 0:
            poss.append(j)
    
    return np.random.choice(poss)




def setup_Agent(filename, epsilon):
    """
    Function to initialize the DQN agent
    """
    # one hot vector (opponents move) on top of game board
    input_dims = 7*7


    action_space = tuple(range(7))
    n_actions = 7

    h1_dims = 512
    h2_dims = 256


    agent = Agent(lr=0.001, gamma=0.95, epsilon=epsilon, epsilon_dec=0.995, epsilon_min=0.01,
                  input_shape=input_dims, h1_dims=h1_dims, h2_dims=h2_dims, action_space=action_space,
                  training_epochs=2, fname=filename
                  )


    memory = ReplayBuffer(50000, input_dims, n_actions)

    return agent, memory




"""
Main training Lopp
"""



if __name__ == '__main__':

    train = True
    load = False

    # True uses player 2 as a random agent rather than a DQN agent
    # Will only store memory and train agent 1
    single_agent = True



    env = ConnectFour()

    filename_1 = 'p1.h5'
    filename_2 = 'p2.h5'

    agent_1, memory_1 = setup_Agent(filename_1, epsilon=1)

    agent_2, memory_2 = setup_Agent(filename_2, epsilon=1)

    if load == True:
        agent_1.load_model(filename_1)

        if single_agent == False:
            agent_2.load_model(filename_2)

        print('\n\n... Model Loaded ...\n\n')




    batch_size = 512
    frame_skips = 2500


    wins = [0, 0, 0]
    p1_wins = []
    p2_wins = []
    draws = []
    moves = []


    p_i, p_syms = 0, ('\\', '|', '/', '-')

    for episode in range(1, 50000):
        
        state = env.reset()

        frame = 0

        while not env.done:
            print('Playing a game...  ' + p_syms[p_i], end='\r')
            p_i = (p_i+1) % 4


            # Choose an action depending on player, loop until valid move selected
            # If valid options are chosen repeatedly, a random valid move
            # maybe add negative reward as well?

            tries = 0
            while True:
                if tries > 100:
                    action = random_Agent(env.board)

                else:
                    if env.player == 1:
                        action, actions = agent_1.choose_action(state)
                    elif env.player == 2:
                        if single_agent == True:
                            action, actions = random_Agent(env.board), list(range(7))
                        action, actions = agent_2.choose_action(state)


                valid, state_, p1_rwd, p2_rwd, winner = env.Step(action)


                # Store memory here to give negative reward for invalid moves
                if env.player == 1 or env.done == True:
                    memory_1.store_transition(state, action, p1_rwd, state_, env.done)
            
                if (env.player == 2 or env.done == True) and single_agent == False:
                    memory_2.store_transition(state, action, p2_rwd, state_, env.done)

                if valid == True:
                    break
                else:
                    tries += 1


            if train == True and memory_1.mem_cntr > frame_skips:
                agent_1.learn(batch_size, memory_1.sample_buffer(batch_size))
                
                if single_agent == False:
                    agent_2.learn(batch_size, memory_2.sample_buffer(batch_size))




            state = state_

            frame += 1

            # Update current player here, easier to separate of board states for each player
            env.update_Players()





        # Update winner tallies
        if winner == 1:
            wins[0] += 1
        elif winner == 2:
            wins[1] += 1
        else:
            wins[2] += 1



        p1_wins.append(wins[0] / episode)
        p2_wins.append(wins[1] / episode)
        draws.append(wins[2])
        moves.append(frame)

        plot_progress(episode, p1_wins, p2_wins, draws, moves, filename='progress.png')

        print('\n\nEpisode', episode, 'complete\n\n')


        if episode % 10 == 0 and train == True:
            agent_1.save_model()

            if single_agent != True:
                agent_2.save_model()

            print('\n... Model Saved ...\n')







