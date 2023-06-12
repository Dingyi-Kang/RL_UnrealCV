import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPooling2D
import gym
import scipy.signal
import time
import argparse, gym_unrealcv
import os

#preprocess function converts it to grayscale, and make its value between -1 and 1 (normalization)
def preprocess(observation):
    new_maxd = 1.0
    new_mind = -1.0
    mx_d = 255.0
    mn_d = 0.0

    obs_range = mx_d - mn_d
    # Clip the observation --- make sure each pixel value is between 0-255
    observation = observation.clip(mn_d, mx_d)
    # Rescale the observation
    new_observation = ((observation - mn_d) * (new_maxd - new_mind) / obs_range) + new_mind
    # converts it to grayscale
    #function np.mean() with axis=2 collapses the third dimension by taking the mean along that axis, resulting in a 2D array
    #so return is in shape of 336x336
    return np.mean(new_observation, axis=2)


def stack_frames(stacked_frames, frame, buffer_size):
    if stacked_frames is None:
        #the * operator can be used to "unpack" the list or tuple into separate arguments.
        #here, frame.shape returns a tuple representing the shape of the frame (for example, (height, width)). So, *frame.shape would unpack this tuple into separate arguments.
        stacked_frames = np.zeros((*frame.shape, buffer_size))

        # For the first observation, we feed in the same frame buffer_size times
        for idx in range(buffer_size):
            stacked_frames[:,:,idx] = frame
    else:
        # Scroll frames to the back, discard the oldest frame
        stacked_frames = np.roll(stacked_frames, shift=-1, axis=-1)
        # Add the new frame to the front
        stacked_frames[:,:,-1] = frame

    return stacked_frames


def discounted_cumulative_sums(x, discount):
    # Discounted cumulative sums of vectors for computing rewards-to-go and advantage estimates
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Buffer:
    # Buffer for storing trajectories
    def __init__(self, observation_dimensions, size, gamma=0.99, lam=0.95):
        # Buffer initialization
        self.observation_buffer = np.zeros(
            (size, *observation_dimensions), dtype=np.float32
        )
        self.action_buffer = np.zeros(size, dtype=np.int32)
        self.advantage_buffer = np.zeros(size, dtype=np.float32)
        self.reward_buffer = np.zeros(size, dtype=np.float32)
        self.return_buffer = np.zeros(size, dtype=np.float32)
        self.value_buffer = np.zeros(size, dtype=np.float32)
        self.logprobability_buffer = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.pointer, self.trajectory_start_index = 0, 0

    def store(self, observation, action, reward, value, logprobability):
        # Append one step of agent-environment interaction
        self.observation_buffer[self.pointer] = observation
        self.action_buffer[self.pointer] = action
        self.reward_buffer[self.pointer] = reward
        self.value_buffer[self.pointer] = value
        self.logprobability_buffer[self.pointer] = logprobability
        self.pointer += 1

    def finish_trajectory(self, last_value=0):
        # Finish the trajectory by computing advantage estimates and rewards-to-go
        path_slice = slice(self.trajectory_start_index, self.pointer)
        rewards = np.append(self.reward_buffer[path_slice], last_value)
        values = np.append(self.value_buffer[path_slice], last_value)

        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]

        self.advantage_buffer[path_slice] = discounted_cumulative_sums(
            deltas, self.gamma * self.lam
        )
        self.return_buffer[path_slice] = discounted_cumulative_sums(
            rewards, self.gamma
        )[:-1]

        self.trajectory_start_index = self.pointer

    def get(self):
        # Get all data of the buffer and normalize the advantages
        self.pointer, self.trajectory_start_index = 0, 0
        advantage_mean, advantage_std = (
            np.mean(self.advantage_buffer),
            np.std(self.advantage_buffer),
        )
        self.advantage_buffer = (self.advantage_buffer - advantage_mean) / advantage_std
        return (
            self.observation_buffer,
            self.action_buffer,
            self.advantage_buffer,
            self.return_buffer,
            self.logprobability_buffer,
        )
#end of the class buffer
def create_cnn(observation_dimensions, num_actions):
    model = keras.models.Sequential()
    model.add(Conv2D(32, (5, 5), strides=1, padding='same', activation='relu', 
                     input_shape=observation_dimensions))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(Conv2D(32, (5, 5), strides=1, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(Conv2D(64, (4, 4), strides=1, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(Conv2D(64, (3, 3), strides=1, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(num_actions, activation='softmax'))

    return model
def logprobabilities(logits, a):
    # Compute the log-probabilities of taking actions a by using the logits (i.e. the output of the actor)
    logprobabilities_all = tf.nn.log_softmax(logits)
    logprobability = tf.reduce_sum(
        tf.one_hot(a, num_actions) * logprobabilities_all, axis=1
    )
    return logprobability


# Sample action from actor
@tf.function
def sample_action(observation):
    logits = actor(observation)
    action = tf.squeeze(tf.random.categorical(logits, 1), axis=1)
    return logits, action


# Train the policy by maxizing the PPO-Clip objective
@tf.function
def train_policy(
    observation_buffer, action_buffer, logprobability_buffer, advantage_buffer
):

    with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
        ratio = tf.exp(
            logprobabilities(actor(observation_buffer), action_buffer)
            - logprobability_buffer
        )
        min_advantage = tf.where(
            advantage_buffer > 0,
            (1 + clip_ratio) * advantage_buffer,
            (1 - clip_ratio) * advantage_buffer,
        )

        policy_loss = -tf.reduce_mean(
            tf.minimum(ratio * advantage_buffer, min_advantage)
        )
    policy_grads = tape.gradient(policy_loss, actor.trainable_variables)
    policy_optimizer.apply_gradients(zip(policy_grads, actor.trainable_variables))

    kl = tf.reduce_mean(
        logprobability_buffer
        - logprobabilities(actor(observation_buffer), action_buffer)
    )
    kl = tf.reduce_sum(kl)
    return kl


# Train the value function by regression on mean-squared error
@tf.function
def train_value_function(observation_buffer, return_buffer):
    with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
        value_loss = tf.reduce_mean((return_buffer - critic(observation_buffer)) ** 2)
    value_grads = tape.gradient(value_loss, critic.trainable_variables)
    value_optimizer.apply_gradients(zip(value_grads, critic.trainable_variables))


# Hyperparameters of the PPO algorithm
steps_per_epoch = 200 #store memories of 200 steps
epochs = 1000
gamma = 0.99
clip_ratio = 0.2
policy_learning_rate = 3e-4
value_function_learning_rate = 1e-3
train_policy_iterations = 80
train_value_iterations = 80
lam = 0.97
target_kl = 0.01

# True if you want to render the environment
render = False
#the frist agent is the tracker
agentIndex = 0
'''may continue training by making it ture'''
load_checkpoint = False
inpSize = 336
parser = argparse.ArgumentParser(description=None)
parser.add_argument("-e", "--env_id", nargs='?', default='UnrealSearch-RealisticRoomDoor-DiscreteColor-v0', #'UnrealArm-DiscreteRgbd-v0', #'RobotArm-Discrete-v0',
                    help='Select the environment to run')
parser.add_argument("-r", '--render', dest='render', action='store_true', help='show env using cv2')
args = parser.parse_args()
env = gym.make(args.env_id, resolution=(inpSize,inpSize))
print("Env created")

num_actions = env.action_space[agentIndex].n
# Initialize the buffer

observation_dimensions = (inpSize, inpSize, 4)
buffer = Buffer(observation_dimensions, steps_per_epoch)

# Initialize the actor and the critic as keras models
observation_input = keras.Input(shape=observation_dimensions, dtype=tf.float32)
actor = create_cnn(observation_dimensions, num_actions)
critic = create_cnn(observation_dimensions, 1)


dir_path = 'tmp2/checkpoints'
f = open('PPO_records.txt', 'a')
# Check if the directory exists
if not os.path.exists(dir_path):
    # If the directory does not exist, create it
    print("!!!Successfully created the directory")
    os.makedirs(dir_path)

actor_checkpoint_file = os.path.join(dir_path, 'actor.ckpt')
critic_checkpoint_file = os.path.join(dir_path, 'critic.ckpt')

if load_checkpoint:
    print('...Loading Checkpoint...')
    actor.load_weights(actor_checkpoint_file)
    critic.load_weights(critic_checkpoint_file)

# Initialize the policy and the value function optimizers
policy_optimizer = keras.optimizers.Adam(learning_rate=policy_learning_rate)
value_optimizer = keras.optimizers.Adam(learning_rate=value_function_learning_rate)



'''Begin to train'''
# Initialize the observation, episode return and episode length
stack_size = 4
observations, episode_return, episode_length = env.reset(), 0, 0
curr_observation = preprocess(observations[agentIndex])
stacked_frames = None
stacked_frames = stack_frames(stacked_frames, curr_observation, stack_size)
# ------ train ------
# Iterate over the number of epochs
for epoch in range(epochs):
    # Initialize the sum of the returns, lengths and number of episodes for each epoch
    sum_return = 0
    sum_length = 0
    num_episodes = 0

    # Iterate over the steps of each epoch
    for t in range(steps_per_epoch):
        if render:
            env.render()

        # Get the logits, action, and take one step in the environment
        reshaped_stacked_frames = stacked_frames.reshape(1, *stacked_frames.shape)
        logits, action = sample_action(reshaped_stacked_frames)
        observations, rewards, done, _ = env.step(action[0].numpy())
        reward = rewards[agentIndex]
        episode_return += reward
        episode_length += 1

        # Get the value and log-probability of the action
        value_t = critic(reshaped_stacked_frames)
        logprobability_t = logprobabilities(logits, action)

        # Store obs, act, rew, v_t, logp_pi_t
        buffer.store(reshaped_stacked_frames, action, reward, value_t, logprobability_t)

        # Update the observation
        curr_observation = preprocess(observations[agentIndex])
        stacked_frames = stack_frames(stacked_frames, curr_observation, stack_size)

        # Finish trajectory if reached to a terminal state
        terminal = done
        if terminal or (t == steps_per_epoch - 1):
            last_value = 0 if done else critic(stacked_frames.reshape(1, *stacked_frames.shape))
            buffer.finish_trajectory(last_value)
            sum_return += episode_return
            sum_length += episode_length
            num_episodes += 1
            observations, episode_return, episode_length = env.reset(), 0, 0
            curr_observation = preprocess(observations[agentIndex])
            stacked_frames = None
            stacked_frames = stack_frames(stacked_frames, curr_observation, stack_size)
        
    print(f"Epoch: {epoch + 1}.")
    # Get values from the buffer
    (
        observation_buffer,
        action_buffer,
        advantage_buffer,
        return_buffer,
        logprobability_buffer,
    ) = buffer.get()

    # Update the policy and implement early stopping using KL divergence
    for _ in range(train_policy_iterations):
        kl = train_policy(
            observation_buffer, action_buffer, logprobability_buffer, advantage_buffer
        )
        if kl > 1.5 * target_kl:
            # Early Stopping
            break

    # Update the value function
    for _ in range(train_value_iterations):
        train_value_function(observation_buffer, return_buffer)

    # Print mean return and length for each epoch
    print(
        f" Epoch: {epoch + 1}. Mean Return: {sum_return / num_episodes}. Mean Length: {sum_length / num_episodes}"
    )
    
    report_line = f"Epoch: {epoch + 1}. Mean Return: {sum_return / num_episodes}. Mean Length: {sum_length / num_episodes}\n"
    f.write(report_line)
    f.flush()

    print('...Saving Checkpoint...')
    actor.save_weights(actor_checkpoint_file)
    critic.save_weights(critic_checkpoint_file)

f.close()