import os
import numpy as np
import tensorflow as tf


class PolicyGradientAgent(object):
    def __int__(self, ALPHA, GAMMA=0.95, n_actions=7, fcl=256, 
                input_shape=(336,336), channels=1, chkpt_dir="tmp/checkpoints",
                gpu={'GPU':1}):
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
        config = tf.ConfigProto(device_count=self.gpu)
        self.sess = tf.Session(config=config)
        self.build_net()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.checkpoint_file = os.path.join(chkpt_dir, 'policy_network.ckpt')
    
    def build_net(self):
        with tf.variable_scope('parameters'):
            self.input = tf.placeholder(tf.float32,
                            shape=[None, self.input_height, self.input_width,
                            self.channels])
            self.label = tf.placeholder(tf.int32, shape=[None], name='label')
            self.G = tf.placeholder(tf.float32, shape=[None,], name='G')

        with tf.variable_scope('conv_layer_1'):
            conv1 = tf.layers.conv2d(input=self.input, filters=32,
                                     kernel_size=(5,5), strides=1, padding='same', name='conv1',
                                     kernel_initializer=tf.contrib.layers.xavier_initializer())
            maxp1 = tf.layers.max_pooling2d(conv1, pool_size=(2,2), strides=2, name='maxp1')
            conv1_activated = tf.nn.relu(maxp1)

        with tf.variable_scope('conv_layer_2'):
            conv2 = tf.layers.conv2d(input=conv1_activated, filters=32,
                                     kernel_size=(5,5), strides=1, padding='same', name='conv2',
                                     kernel_initializer=tf.contrib.layers.xavier_initializer())
            maxp2 = tf.layers.max_pooling2d(conv2, pool_size=(2,2), strides=2, name='maxp2')
            conv2_activated = tf.nn.relu(maxp2)

        with tf.variable_scope('conv_layer_3'):
            conv3 = tf.layers.conv2d(inputs=conv2_activated, filters=64,
                                     kernel_size=(4,4), strides=1, padding='same', name='conv3',
                                     kernel_initializer=tf.contrib.layers.xavier_initializer())
            maxp3 = tf.layers.max_pooling2d(conv3, pool_size=(2,2), strides=2, name='maxp3')
            conv3_activated = tf.nn.relu(maxp3)

        with tf.variable_scope('conv_layer_4'):
            conv4 = tf.layers.conv2d(inputs=conv3_activated, filters=64,
                                     kernel_size=(3,3), strides=1, padding='same', name='conv4',
                                     kernel_initializer=tf.contrib.layers.xavier_initializer())
            maxp4 = tf.layers.max_pooling2d(conv4, pool_size=(2,2), strides=2, name='maxp4')
            conv4_activated = tf.nn.relu(maxp4)

        with tf.variable_scope('fc1'):
            flat = tf.layers.flatten(conv4_activated)
            dense1 = tf.layers.dense(flat, units=self.fcl, activation=tf.nn.relu)

        with tf.variable_scope('fc2'):
            dense2 = tf.layers.dense(dense1, units=self.n_actions,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer())
        
        self.actions = tf.nn.softmax(dense2, name='actions')

        with tf.variable_scope('loss'):
            negative_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(
                                        logits=dense2, labels=self.label)
            self.loss = tf.reduce_mean(negative_log_prob * self.G)

        with tf.variable_scope('train'):
            self.train_op = tf.train.RMSPropOptimizer(self.lr, decy=0.99,
                                momentum=0.0, epsilon=1e-6).minimize(self.loss)

    def choose_action(self, observation):
        observation = np.array(observation).reshape((-1, self.input_height,
                                                        self.input_width,
                                                        self.channels))

        probabilities = self.sess.run(self.actions, feed_dict={self.input: observation})[0]

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
                # if t > 0 and reward_memory[k] - reward_memory[k-1] < 0:
                #     break
                G_sum += reward_memory[k] * discount
                discount *= self.gamma
            G[t] = G_sum

        mean = np.mean(G)
        std = np.std(G) if np.std(G) > 0 else 1
        G = (G - mean) / std

        _ = self.sess.run(self.train_op,
                            feed_dict={self.input:state_memory,
                                       self.label:action_memory,
                                       self.G:G})

        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []

    def load_checkpoint(self):
        print('...Loading Checkpoint...')
        self.saver.restore(self.sess, self.checkpoint_file)

    def save_checkpoint(self):
        print('...Saving Checkpoint...')
        self.saver.save(self.sess, self.checkpoint_file)




