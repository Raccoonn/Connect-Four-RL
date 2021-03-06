

import matplotlib.pyplot as plt
import numpy as np





class ReplayBuffer(object):
    """
    Memory buffer used to store states and sample during training
    """
    def __init__(self, max1_size, input_shape, n_actions, discrete=True):
        self.mem_size = max1_size
        self.mem_cntr = 0
        self.discrete = discrete
        self.state_memory = np.zeros((self.mem_size, input_shape))
        self.state__memory = np.zeros((self.mem_size, input_shape))
        dtype = np.int8 if self.discrete else np.float32
        self.action_memory = np.zeros((self.mem_size, n_actions), dtype=dtype)
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)


    def store_transition(self, state, action, reward, state_, done):
        """
        Store values for current and next state
        """
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.state__memory[index] = state_
        
        # store one hot encoding of actions, if appropriate
        if self.discrete:
            actions = np.zeros(self.action_memory.shape[1])
            actions[action] = 1.0
            self.action_memory[index] = actions
        else:
            self.action_memory[index] = action

        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - done
        self.mem_cntr += 1


    def sample_buffer(self, batch_size):
        """
        Draw a sample from the store memory of states, actions, and rewards
        """
        max1_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max1_mem, batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.state__memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal




def plot_progress(episode, p1_wins, p2_wins, draws, moves, filename='progress'):
    """
    Plot progress and save to file during trtaining
    """
    x = list(range(episode))
    
    fig1, ax1 = plt.subplots()

    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Game Wins')

    line1, = ax1.plot(x, p1_wins)
    line2, = ax1.plot(x, p2_wins)
    # line3, = ax1.plot(x, draws)

    plt.legend((line1, line2), ('Player 1', 'Player 2'),
               loc='upper left')

    plt.savefig(filename + '.png')

    plt.close()


    fig2, ax2 = plt.subplots()

    ax2.set_xlabel('Episodes')
    ax2.set_ylabel('Moves per Episode')
    line4, = ax2.plot(x, moves, '.')

    plt.savefig(filename + '_moves.png')

    plt.close()

