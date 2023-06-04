import numpy as np
import gym
from reinforce_cnn_tf import PolicyGradientAgent
from utils import plotLearning
from gym import wrappers
import argparse, gym_unrealcv
import os
import cv2

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('tracker_view.mp4', fourcc, 20.0, (336, 336))

#preprocess function converts it to grayscale, and make its value between -1 and 1
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
    return np.mean(new_observation, axis=2)

def stack_frames(stacked_frames, frame, buffer_size):
    if stacked_frames is None:
        #the * operator can be used to "unpack" the list or tuple into separate arguments.
        #here, frame.shape returns a tuple representing the shape of the frame (for example, (height, width)). So, *frame.shape would unpack this tuple into separate arguments.
        stacked_frames = np.zeros((buffer_size, *frame.shape))
        for idx, _ in enumerate(stacked_frames):
            stacked_frames[idx,:] = frame
    else:
        stacked_frames[0:buffer_size-1,:] = stacked_frames[1:,:]
        stacked_frames[buffer_size-1, :] = frame

    return stacked_frames

def stack_actions(stacked_actions, action, buffer_size):
    if stacked_actions is None:
        #the * operator can be used to "unpack" the list or tuple into separate arguments.
        #here, frame.shape returns a tuple representing the shape of the frame (for example, (height, width)). So, *frame.shape would unpack this tuple into separate arguments.
        stacked_actions = np.zeros((buffer_size, *action.shape))
        for idx, _ in enumerate(stacked_actions):
            stacked_actions[idx,:] = action
    else:
        stacked_actions[0:buffer_size-1,:] = stacked_actions[1:,:]
        stacked_actions[buffer_size-1, :] = action

if __name__ == '__main__':
    '''may continue training by making it ture'''
    load_checkpoint = False
    dir_path = 'tmp/checkpoints'
    # Check if the directory exists
    if not os.path.exists(dir_path):
        # If the directory does not exist, create it
        print("!!!Successfully created the directory")
        os.makedirs(dir_path)

    inpSize = 336
    #num_channels = 3
    stack_size = 5
    agent = PolicyGradientAgent(ALPHA=0.0001, GAMMA=0.9, n_actions=7, sequence_size = stack_size, lstm_neurons=256, input_shape=(inpSize, inpSize), channels=1, 
                                chkpt_dir=dir_path, gpu={'GPU':0})
    filename = 'score_history_policyGradient.png'
    print('will use ', filename, ' and ', agent.gpu)
    if load_checkpoint:
        agent.load_checkpoint()
       
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument("-e", "--env_id", nargs='?', default='UnrealSearch-RealisticRoomDoor-DiscreteColor-v0', #'UnrealArm-DiscreteRgbd-v0', #'RobotArm-Discrete-v0',
                        help='Select the environment to run')
    parser.add_argument("-r", '--render', dest='render', action='store_true', help='show env using cv2')
    args = parser.parse_args()
    env = gym.make(args.env_id, resolution=(inpSize,inpSize))
    #env.seed(1)
    print("Env created")

    score_history = []
    score = 0
    num_episodes = 3000
    #by far, looks like index of 0 corresponds to the ob and reward of tracker
    agentIndex = 0
    for i in range(num_episodes):
        done = False
        observations = env.reset()
        # env.render()
        # print(observations.shape) #(2, 336, 336, 3)
        ## use only the view of the tracker
        curr_observation = preprocess(observations[agentIndex]) #print('Shape of preprocessedObservation: ', observation.shape)
        stacked_frames = None
        stacked_frames = stack_frames(stacked_frames, curr_observation, stack_size)
        action = agent.choose_action(stacked_frames)
        #stacked_actions = None
        #stacked_actions = stack_frames(stacked_actions, action, stack_size)
        score = 0
        while not done:
            observations, rewards, done, info = env.step(action)
            #print('Shape of rewards: ', rewards.shape) #-->(2,)
            reward = rewards[agentIndex]
            agent.store_transition(stacked_frames, action, reward)

            #cv2.imwrite('tracker_view.png', cv2.cvtColor(observations[agentIndex], cv2.COLOR_RGB2BGR))
            #only record first and last 100 episodes
            if i < 100 or i >= num_episodes - 100:
              out.write(observations[agentIndex])

            curr_observation = preprocess(observations[agentIndex])
            stacked_frames = stack_frames(stacked_frames, curr_observation, stack_size)
            action = agent.choose_action(stacked_frames)
            #stacked_actions = stack_frames(stacked_actions, action, stack_size)

            score += reward
        
        score_history.append(score)

        if i % 20 == 19:
            avg_score = np.mean(score_history[max(0, i-20):(i+1)])
            print('episode: ', i,'score: ', score, ' average score %.3f' % avg_score)
            plotLearning(score_history, filename=filename, window=20)
        else:
            print('episode: ', i,'score: ', score)

        
        agent.learn()

        if i % 20 == 19:
            agent.save_checkpoint()
    plotLearning(score_history, filename=filename, window=20)
    env.close()
    out.release()
    print("end")
