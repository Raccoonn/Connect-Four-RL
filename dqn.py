
import tensorflow as tf
import numpy as np



"""
    - Changed relu activation to LeakyReLU layers

    - Changed input to 7*7, one hot for opponent move then game board


"""






class Agent:
    def __init__(self, lr, gamma, epsilon, epsilon_dec, epsilon_min,
                 input_shape, h1_dims, h2_dims, action_space,
                 training_epochs=1, fname='model.h5'
                ):
        """
        Initialize network with given hyperparameters
        """
        
        # Define parameters
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_min

        self.input_shape = input_shape
        self.h1_dims = h1_dims
        self.h2_dims = h2_dims 

        self.action_space = action_space
        self.n_actions = len(action_space)

        # Used to train for multiple epochs per training batch
        self.training_epochs = training_epochs


        # Set filename for saving network
        self.fname = fname


        # Compile TF network
        self.q_eval = tf.keras.models.Sequential([
                      tf.keras.layers.Dense(h1_dims, input_shape=(input_shape,)),
                      tf.keras.layers.LeakyReLU(alpha=0.3),
                      tf.keras.layers.Dense(h2_dims),
                      tf.keras.layers.LeakyReLU(alpha=0.3),
                      tf.keras.layers.Dense(self.n_actions)
            ])

        self.q_eval.compile(optimizer=tf.keras.optimizers.Adam(lr=lr), loss='mse')




    def choose_action(self, state):
        """
        Given a state choose next action
        """
        state = state[np.newaxis, :]

        # Choose random action if epsilon reached
        if np.random.random() < self.epsilon:
            poss = []
            for j in range(7):
                if state[0,j] == 0:
                    poss.append(j)
            action, actions = np.random.choice(poss), poss        
        # Else preform forward pass using state input
        else:
            actions = self.q_eval.predict(state)
            action = np.argmax(actions)

        return action, np.around(actions, decimals=2)




    def learn(self, batch_size, memory_sample):
        """
        Given a memory sample preform a training loop
        """

        state, action, reward, state_, done = memory_sample

        action_values = np.array(list(range(self.n_actions)), dtype=np.int8)
        action_indices = np.dot(action, action_values)
        q_eval = self.q_eval.predict(state)

        q_next = self.q_eval.predict(state_)

        q_target = q_eval.copy()

        batch_index = np.arange(batch_size, dtype=np.int32)

        q_target[batch_index, action_indices] = reward + self.gamma*np.max(q_next, axis=1)*done


        # Train batch for specified Epochs
        for _ in range(self.training_epochs):
            history = self.q_eval.fit(state, q_target, verbose=0)


        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_dec
        else:
            self.epsilon = self.epsilon_min

        return history







    def save_model(self):
        """
        Save model to file
        """
        self.q_eval.save(self.fname)



    def load_model(self, fname):
        """
        Load model from file
        """
        self.q_eval = tf.keras.models.load_model(fname)
        self.q_eval.compile(optimizer=tf.keras.optimizers.Adam(lr=self.lr), loss='mse')

