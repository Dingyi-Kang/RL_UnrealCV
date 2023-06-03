import os
import numpy as np
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import LSTM, TimeDistributed

class PolicyGradientAgent(object):
    def __init__(self, ALPHA, GAMMA=0.95, n_actions=7, lstm_neurons=256, 
                 input_shape=(336, 336), channels=1, chkpt_dir="tmp/checkpoints",
                 gpu={'GPU': 1}):
        self.lr = ALPHA
        self.gamma = GAMMA
        self.n_actions = n_actions
        self.action_space = [i for i in range(n_actions)]
        self.input_height = input_shape[0]
        self.input_width = input_shape[1]
        self.channels = channels
        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []
        self.gpu = gpu
        self.lstm_neurons = lstm_neurons
        self.model = None
        self.build_net()
        self.model.compile(loss='sparse_categorical_crossentropy',
                           optimizer=RMSprop(lr=self.lr), sample_weight_mode='temporal')
        self.checkpoint_file = os.path.join(chkpt_dir, 'policy_network.ckpt')
    
    def build_net(self):
        self.model = Sequential([
            TimeDistributed(Conv2D(filters=32, kernel_size=(5, 5), strides=1, padding='same',
                   activation='relu'), input_shape=(None, self.input_height, self.input_width, self.channels)),
            TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=2)),
            TimeDistributed(Conv2D(filters=32, kernel_size=(5, 5), strides=1, padding='same',
                   activation='relu')),
            TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=2)),
            TimeDistributed(Conv2D(filters=64, kernel_size=(4, 4), strides=1, padding='same',
                   activation='relu')),
            TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=2)),
            TimeDistributed(Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same',
                   activation='relu')),
            TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=2)),
            TimeDistributed(Flatten()),
            LSTM(units=self.lstm_neurons, activation='tanh', return_sequences=True),
            TimeDistributed(Dense(units=self.n_actions, activation='softmax'))
        ])
        
    def choose_action(self, observation):
        observation = np.array(observation).reshape((1, 4, self.input_height,
                                                     self.input_width,
                                                     self.channels))
        
        #one layer is batch size (only one batch); second layer is time sequence (we use the last one)
        probabilities = self.model.predict(observation)[0][-1]
        #print(probabilities)
        action = np.random.choice(self.action_space, p=probabilities)

        return action
    
    def store_transition(self, observation, action, reward):
        self.state_memory.append(observation)
        self.action_memory.append(action)
        self.reward_memory.append(reward)

    def learn(self):
        seqLength = len(self.state_memory)
        state_memory = np.array(self.state_memory).reshape(
            (1, seqLength, self.input_height, self.input_width, self.channels))
        #3D is required. the second dimension is time sequence
        action_memory = np.array(self.action_memory).reshape(1, seqLength, 1)
        reward_memory = np.array(self.reward_memory)

        G = np.zeros_like(reward_memory)
        for t in range(len(reward_memory)):
            G_sum = 0
            discount = 1
            for k in range(t, len(reward_memory)):
                G_sum += reward_memory[k] * discount
                discount *= self.gamma
            G[t] = G_sum

        mean = np.mean(G)
        std = np.std(G) if np.std(G) > 0 else 1
        G = (G - mean) / std

        G = G.reshape(1, seqLength)
        self.model.fit(state_memory, action_memory, sample_weight=G, verbose=0)

        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []

    def load_checkpoint(self):
        print('...Loading Checkpoint...')
        self.model.load_weights(self.checkpoint_file)

    def save_checkpoint(self):
        print('...Saving Checkpoint...')
        self.model.save_weights(self.checkpoint_file)
