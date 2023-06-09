import os
import numpy as np
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop


class PolicyGradientAgent(object):
    def __init__(self, ALPHA, GAMMA=0.95, n_actions=7, fcl=256, 
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
        self.fcl = fcl
        self.model = None
        self.build_net()
        self.model.compile(loss='sparse_categorical_crossentropy',
                           optimizer=RMSprop(lr=self.lr))
        self.checkpoint_file = os.path.join(chkpt_dir, 'policy_network.ckpt')
    
    def build_net(self):
        self.model = Sequential([
            Conv2D(filters=32, kernel_size=(5, 5), strides=1, padding='same',
                   activation='relu', input_shape=(self.input_height, self.input_width, self.channels)),
            MaxPooling2D(pool_size=(2, 2), strides=2),
            Conv2D(filters=32, kernel_size=(5, 5), strides=1, padding='same',
                   activation='relu'),
            MaxPooling2D(pool_size=(2, 2), strides=2),
            Conv2D(filters=64, kernel_size=(4, 4), strides=1, padding='same',
                   activation='relu'),
            MaxPooling2D(pool_size=(2, 2), strides=2),
            Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same',
                   activation='relu'),
            MaxPooling2D(pool_size=(2, 2), strides=2),
            Flatten(),
            Dense(units=self.fcl, activation='relu'),
            Dense(units=self.n_actions, activation='softmax')
        ])
        
    def choose_action(self, observation):
        observation = np.array(observation).reshape((-1, self.input_height,
                                                     self.input_width,
                                                     self.channels))

        probabilities = self.model.predict(observation)[0]

        action = np.random.choice(self.action_space, p=probabilities)

        return action
    
    def store_transition(self, observation, action, reward):
        self.state_memory.append(observation)
        self.action_memory.append(action)
        self.reward_memory.append(reward)

    def learn(self):
        state_memory = np.array(self.state_memory).reshape(
            (-1, self.input_height, self.input_width, self.channels))
        action_memory = np.array(self.action_memory)
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
