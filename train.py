from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.tools import inspect_checkpoint as chkp
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes

import matplotlib.pyplot as plt
import numpy as np
import os
from os import listdir
from os.path import isfile, join, isdir
from random import shuffle, choice
from PIL import Image
import sys
import json
import collections, cv2

import argparse, gym_unrealcv, gym, time, imageio
from gym import wrappers
from scipy import ndimage
import scipy as sp

def warn(*args, **kwargs):
	pass

#only messages with severity ERROR or higher will be shown. Lower-severity messages (like WARN and INFO) will be suppressed. 
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = "true" 

# tf.set_random_seed(0)

# input_width = 224
# input_height = 224
# num_channels = 3
slim = tf.contrib.slim
n_hidden1 = 512
n_hidden2 = 512
learnError = 0
n_epochs = 1
batch_size = 2

feature_size = 512
seq_length = batch_size - 1
statefull_lstm_flag = False
lr_init = 1e-12

inpSize = input_width = input_height = 336
num_channels = 3
attnSize = 21
attention_shape = attnSize*attnSize

parser = argparse.ArgumentParser(description=None)
parser.add_argument("-e", "--env_id", nargs='?', default='RobotArm-Discrete-v0',
					help='Select the environment to run')
parser.add_argument("-r", '--render', dest='render', action='store_true', help='show env using cv2')
args = parser.parse_args()
env = gym.make(args.env_id, resolution=(inpSize,inpSize))

num_control = 7#env.action_space.n

print(num_control)

def broadcast(tensor, shape):
	return tensor + tf.zeros(shape, dtype=tensor.dtype)

def RNNCell(W, B, inputs, state):
	"""Most basic RNN: output = new_state = act(W * input + U * state + B)."""
	one = constant_op.constant(1, dtype=dtypes.int32)
	add = math_ops.add
	multiply = math_ops.multiply
	sigmoid = math_ops.sigmoid
	activation = math_ops.tanh

	gate_inputs = math_ops.matmul(array_ops.concat([inputs, state], 1), W)
	gate_inputs = nn_ops.bias_add(gate_inputs, B)
	output = sigmoid(gate_inputs)
	return output, output

def lstm_cell(W, b, forget_bias, inputs, state):
	one = constant_op.constant(1, dtype=dtypes.int32)
	add = math_ops.add
	multiply = math_ops.multiply
	sigmoid = math_ops.sigmoid
	activation = math_ops.sigmoid
	# activation = math_ops.tanh

	c, h = array_ops.split(value=state, num_or_size_splits=2, axis=one)

	gate_inputs = math_ops.matmul(array_ops.concat([inputs, h], 1), W)
	gate_inputs = nn_ops.bias_add(gate_inputs, b)
	# i = input_gate, j = new_input, f = forget_gate, o = output_gate
	i, j, f, o = array_ops.split(value=gate_inputs, num_or_size_splits=4, axis=one)

	forget_bias_tensor = constant_op.constant(forget_bias, dtype=f.dtype)

	new_c = add(multiply(c, sigmoid(add(f, forget_bias_tensor))), multiply(sigmoid(i), activation(j)))
	new_h = multiply(activation(new_c), sigmoid(o))
	new_state = array_ops.concat([new_c, new_h], 1)

	return new_h, new_state

class RNN_Predictor (tf.keras.Model):
	def __init__(self, batch_input_shape, input_shape, hidden_units):
	  
		super(RNN_Predictor, self).__init__()
	  
		# hidden state
		self.hidden_units = hidden_units
		
		self.hidden_state = tf.zeros((attention_shape, self.hidden_units))
		
		# BahdanauAttention layer to process input
		self.W1 = tf.keras.layers.Dense(hidden_units) 
		
		# BahdanauAttention layer to process hidden
		self.W2 = tf.keras.layers.Dense(hidden_units)

		# BahdanauAttention layer to generate context vector from embedded input and hidden state
		self.V = tf.keras.layers.Dense(1)

		# Could not find how to choose the "teacher_force" option for the LSTM, I am assuming
		# that is how it generates the hidden states over the sequence.
		# 
		# We also use a grid of 64 LSTMs to process the indiviual image blocks. The weights
		# of the LSTMs are the same. We implement this by passing the 64 inputs as
		# a "batch" into a single LSTM, so the batch size is not the default "BATCH_SIZE", which is 1
		# Note LSTM expects a 3D tensor (batch_size, seq_length, feature)
		self.LSTM1 = tf.keras.layers.LSTM(self.hidden_units,
										 batch_input_shape= (seq_length, batch_input_shape, input_shape),
										 time_major = True, #(timesteps, batch, ...)
										 # input_shape= (seq_length, input_shape),
										 return_sequences=True,
										 return_state=True, # return hidden state
										 stateful= statefull_lstm_flag)
		
		self.LSTM2 = tf.keras.layers.LSTM(self.hidden_units,
										 batch_input_shape= (seq_length, batch_input_shape, input_shape),
										 time_major = True, #(timesteps, batch, ...)
										 # input_shape= (seq_length, input_shape),
										 return_sequences=True,
										 return_state=True, # return hidden state
										 stateful= statefull_lstm_flag) 
		
		self.LSTM3 = tf.keras.layers.LSTM(self.hidden_units,
										 batch_input_shape= (seq_length, batch_input_shape, input_shape),
										 time_major = True, #(timesteps, batch, ...)
										 # input_shape= (seq_length, input_shape),
										 return_sequences=True,
										 return_state=True, # return hidden state
										 stateful= statefull_lstm_flag)	   
		
	def call(self, x): 
		# Dimension of x is [seq_length, attention_shape=64,  hidden_units] and is hidden_state is [attention_shape=64, hidden_units]

		# print ('In RNN Decoder x=', x.shape, 'hidden=', hidden_state.shape)

		# Bahadanou attention -- note we have hooks for temporal attention too.
				
		if (len(self.hidden_state.shape) == 2):
		  hidden_with_time_axis = tf.expand_dims(self.hidden_state, 0)
		else :
		  print ('Error: hidden_state should be 2 dimensional. It is:', hidden_state.shape)
		  hidden_with_time_axis = self.hidden_state
		
		# score shape == (seq_length, attention_shape=64, hidden_units)
		score = tf.nn.tanh(self.W1(x) + self.W2(hidden_with_time_axis))
		
		# attention_weights shape == (seq_length, attention_shape=64, 1)
		# you get 1 at the last axis because you are applying score to self.V
		attention_weights = tf.nn.softmax(self.V(score), axis=1)
		
		# context_vector shape == (seq_length, attention_shape=64, hidden_units)
		context_vector = tf.multiply(x, attention_weights)
		
		#print ('context_vector=', context_vector.shape, 'attention_weights=', attention_weights.shape)
		
		# option 1: concatenate context vector with input x - very large state vector
		# shape after concatenation == (attention_shape=64, 1, embedding_dim + hidden_units)	
		# x_hat = tf.concat([context_vector, x], axis=-1)
		
		x_hat = context_vector  # (seq_length, attention_shape=64, embedding_dim)

		# LSTM expects a 3D tensor (seq_length, batch_size,  feature) since time_major == True
		# we use batch_size = 64 -- the 64 blocks of inceptionV3 features
		# seq_length, and feature = 2048 inceptionV3 features

		#  output, next_hidden_state, cell_state = self.LSTM (x_hat)
	 
		encoder_output1, _ , _ = self.LSTM1 (x_hat)
		encoder_output2, _ , _ = self.LSTM2 (encoder_output1)
		output, next_hidden_state, cell_state = self.LSTM3 (encoder_output2)

		# model.add(LSTM(100, activation='relu', input_shape=(n_in,1)))
		# model.add(RepeatVector(n_in))
		# model.add(LSTM(100, activation='relu', return_sequences=True))
		# model.add(TimeDistributed(Dense(1)))
		# model.compile(optimizer='adam', loss='mse')

		self.hidden_state = next_hidden_state

		#print ('x =', x_hat.shape, 'output=', output.shape, 'h state=', next_hidden_state.shape, 'cell state=', cell_state.shape)			   
			   
		return output, self.hidden_state, attention_weights

 
	def reset_hidden_state(self):
		self.hidden_state = tf.zeros((attention_shape, self.hidden_units))
		# LSTM expects a 3D tensor (batch_size, seq_length, feature)
		# we use batch_size = 64 (attention_shape) -- the 64 blocks of inceptionV3 features
		# seq_length, and feature = 2048 inceptionV3 features
		# hidden state dimension is (batch_size=attention_shape, features)

learnError = 0
@tf.function
def loss_function(target, pred, input_seq):  #Dimesions: [seq_length, 64, 2048]
  
	pred_loss = tf.square(tf.subtract(target, pred))
	
	# if we were to just do zero order hold as prediction, this is the loss or error
	frame_diff = tf.square(tf.subtract(target, input_seq)) 
	
	# prediction loss weighted by frame diffeence (zero-order hold difference), errors will be
	# weighted high for blocks with motion/change
	weighted_loss = tf.multiply(frame_diff, pred_loss) 

	sseLoss = tf.reduce_mean(weighted_loss)/(attnSize*attnSize*512)

	# weighted_loss = pred_loss

	# sseLoss = tf.reduce_mean(weighted_loss)#/(14*14*512)
	
	lossGrid = tf.nn.softmax(tf.reduce_mean(weighted_loss, 2), axis=1)
	
	return sseLoss, lossGrid

scope = 'vgg_16'
fc_conv_padding = 'VALID'
dropout_keep_prob=0.7
is_training = False

with tf.device('/device:GPU:0'):
	inputs = tf.placeholder(tf.float32, (2, inpSize, inpSize, 3), name='inputs')
	targets = tf.placeholder(tf.float32, (1, attnSize, attnSize, 1), name='targets')
	ctrl_targets = tf.placeholder(tf.float32, (1, 1), name='ctrl_targets')

	learning_rate = tf.placeholder(tf.float32, [])

	r, g, b = tf.split(axis=3, num_or_size_splits=3, value=inputs * 255.0)
	VGG_MEAN = [103.939, 116.779, 123.68]
	VGG_inputs = tf.concat(values=[b - VGG_MEAN[0], g - VGG_MEAN[1], r - VGG_MEAN[2]], axis=3)
#### what is the usage of VGG_inputs????? it is not used
	with tf.variable_scope(scope, 'vgg_16', [VGG_inputs]) as sc:
		end_points_collection = sc.original_name_scope + '_end_points'
		# Collect outputs for conv2d, fully_connected and max_pool2d.
		with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
												outputs_collections=end_points_collection):
			net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1', trainable=False)
			net = slim.max_pool2d(net, [2, 2], scope='pool1') #112
			net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2', trainable=False)
			net = slim.max_pool2d(net, [2, 2], scope='pool2') #56
			net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3', trainable=False)
			net = slim.max_pool2d(net, [2, 2], scope='pool3') #28
			net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4', trainable=False)
			net = slim.max_pool2d(net, [2, 2], scope='pool4') #14
			conv4 = net
			print(conv4.shape)
			# sys.exit(0)
			# net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5', trainable=True)
			# net = slim.max_pool2d(net, [2, 2], scope='pool5') #7
			# conv5 = net
			# # Use conv2d instead of fully_connected layers.
			# net = slim.conv2d(net, 4096, [7, 7], padding=fc_conv_padding, scope='fc6', trainable=True)
			# net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
			# 									 scope='dropout6')
			# net = slim.conv2d(net, 4096, [1, 1], scope='fc7', trainable=True)
			# vgg16_Features = tf.reshape(net, (-1,4096))
			# # Convert end_points_collection into a end_point dict.
			# end_points = slim.utils.convert_collection_to_dict(end_points_collection)

	vgg16_Features = tf.reshape(conv4, (-1, attention_shape, 512))
	# Setup LSTM

	predictModel = RNN_Predictor (batch_input_shape=attention_shape, input_shape=feature_size, hidden_units=n_hidden1)
    #So vgg16_Features[0,...] is selecting the entire data slice in all dimensions at the 0th index of the first dimension of the vgg16_Features tensor.
	input_sequence = tf.reshape(vgg16_Features[0,...], (-1, attention_shape, 512))
	predictions, hidden, attention_weights = predictModel(input_sequence)

	p_feats = tf.reshape(tf.keras.layers.GlobalAveragePooling1D()(attention_weights * predictions), (1, 1, -1))
	print(hidden.shape)
	# control = tf.keras.layers.Dense(num_control)(p_feats)
	control1, _, _ = tf.keras.layers.LSTM(n_hidden1, batch_input_shape= (1, attention_shape, feature_size), time_major = True, return_sequences=True, return_state=True, stateful= statefull_lstm_flag)(p_feats)
	print("C1", control1.shape)
	control = tf.keras.layers.Dense(num_control)(tf.reshape(control1, (-1, n_hidden1)))
	print("C", control.shape)

	loss1, loss_grid = loss_function(tf.reshape(conv4[1,...], (-1, attention_shape, 512)), predictions, input_sequence)

	W1 = tf.keras.layers.Dense(attention_shape)
	W2 = tf.keras.layers.Dense(attention_shape)
	pred = tf.nn.tanh(W1(control) + W2(loss_grid))
	print(pred.shape)
	loss2 = tf.reduce_sum(tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, name='sparse_categorical_crossentropy')(ctrl_targets, control))
	loss = loss1 + 0.01*loss2 #+ loss3

# Optimization
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

#####################
### Training loop ###
#####################

init = tf.global_variables_initializer()

saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="vgg_16"))


config = tf.ConfigProto()
config.gpu_options.allow_growth=True

with tf.Session(config=config) as sess:
	# Initialize parameters
	sess.run(init)
	saver.restore(sess, "./vgg_16.ckpt")
	saver = tf.train.Saver(max_to_keep=0)
	avgPredError = 1.0
	outputFile = "./log_predLearn.txt"
	with open(outputFile, 'w') as of:
		of.write("epoch\treward\tlength\n")
	### In case of interruption, load parameters from the last iteration (ex: 29)
	# saver.restore(sess, '/home/saakur/Desktop/ActiveVision/gym-unrealcv/predLearnTrack/Models/SingleAttn/Model_Epoch_225')
	### And update the loop to account for the previous iterations
	#for i in range(29,n_epochs):
	step = 0

	episode_count = 100000
	reward = 0
	done = False

	# LSTM
	new_state = np.random.uniform(-0.5,high=0.5,size=(1, attention_shape, 2*n_hidden1))

	z = np.zeros((attnSize, attnSize))

	try:
		l1 = int(float(attnSize/2))
		l2 = int(float(attnSize/2))
		if l1 >= attnSize:
			l1 = attnSize - 1
		if l2 >= attnSize:
			l2 = attnSize - 1
		z[l1][l2] = 1
	except Exception as e:
		print (e)
		sys.exit(0)

	z = ndimage.filters.gaussian_filter(z, attnSize)

	z = z - np.min(z)
	z = z / np.max(z)

	ideal_grid = np.reshape(z, (1, attnSize, attnSize, 1))

	predError = collections.deque(maxlen=30)

	for i in range(episode_count):
		env.seed(1)
		obs = env.reset()
		obs_init = env.render()
		count_step = 0
		t0 = time.time()
		reward_episode = []
		
		new_state = np.random.uniform(-0.5,high=0.5,size=(1, attention_shape, 2*n_hidden1))
		
		# Run 1 epoch

		lr = lr_init

		avgPredError = 0

		obs_new = np.zeros(obs.shape)
		
		tgt_act = np.reshape(np.random.randint(0, num_control, size=(1,1)), (-1,1))

		with imageio.get_writer('./detect_%d.gif'%(i%5), mode='I') as writer:
			while True:
				#####why need this code below? redundant
				inp = np.reshape(np.vstack([obs, obs_new]), (-1, inpSize, inpSize, 3))
				obs_new, reward, done, _ = env.step([tgt_act[0][0]])
				count_step += 1
				reward_episode.append(reward)
				inp = np.reshape(np.vstack([obs, obs_new]), (-1, inpSize, inpSize, 3))
				ret = sess.run([control, loss, train_op, loss_grid], feed_dict = {inputs: inp, learning_rate:lr, targets:ideal_grid, ctrl_targets:tgt_act})

				# PID comes here
				a = np.reshape(ret[-1], (-1, attnSize, attnSize, 1))
				attnVal = list(np.unravel_index(np.argmax(a, axis=None), a.shape))
				gridW, gridH = int(inpSize/attnSize), int(inpSize/attnSize)
				predY = attnVal[1]*gridH + int(0.5*gridH)
				predX = attnVal[2]*gridW + int(0.5*gridW)

				
				_, ideal_y, ideal_x, _ = list(np.unravel_index(np.argmax(ideal_grid, axis=None), ideal_grid.shape))
				_, curr_y, curr_x, _ = attnVal

				# 0 -> Move forward fast
				# 1 -> Move forward slow
				# 2 -> Move right forward
				# 3 -> Move left forward
				# 4 -> Turn right hard
				# 5 -> Turn left hard
				# 6 -> Stop

				action_labels = ['FwdFast', 'FwdSlw', 'RgtFwd', 'LftFwd', 'RgtHrd', 'LftHrd', 'Stay']
				if ideal_y - curr_y > 0 and 7 > ideal_x - curr_x > 0:
					tgt_act[0,0] = 3
				elif ideal_y - curr_y > 0 and -7 < ideal_x - curr_x < 0:
					tgt_act[0,0] = 2
				elif ideal_x - curr_x >= 7:
					tgt_act[0,0] = 5
				elif ideal_x - curr_x <= -7:
					tgt_act[0,0] = 4
				elif ideal_y - curr_y >= 14:
					# if np.random.uniform() > 0.95:
					# 	tgt_act[0,0] = 6
					# else:	
					tgt_act[0,0] = 0
				elif ideal_y - curr_y < 14:
					tgt_act[0,0] = 1
				else:
					tgt_act[0,0] = 6

				ret1 = sess.run([train_op], feed_dict = {inputs: inp, learning_rate:lr, targets:ideal_grid, ctrl_targets:tgt_act})

				print("\r Step:%d"%count_step, " Reward:", reward, "Loss:", np.mean(predError), end="")
				img = obs[..., ::-1].copy()  # bgr->rgb
				
				cv2.rectangle(img, (predX - 25, predY - 25), (predX + 25, predY + 25), (0, 0, 255), -1, cv2.LINE_AA)
				# cv2.putText(img, str(reward), (inpSize - 100, inpSize - 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
				# cv2.putText(img, str(action_labels[action]) + "," + str(action_labels[tgt_act[0][0]]), (25, inpSize - 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
				# cv2.putText(img, str(ret[1]), (inpSize - 100, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
				writer.append_data(img)
				if done:
					with open(outputFile, 'a') as of:
						of.write("%d\t%f\t%d\n"%(i, np.sum(reward_episode), count_step))
					fps = count_step / (time.time() - t0)
					print ('Fps:' + str(fps) + ' Reward:' + str(np.sum(reward_episode)) + " E_L: " + str(count_step))
					reward_episode = []
					break
				predError.append(ret[1])
				avgPredError = np.mean(predError)
				obs = obs_new

		# if (i+1) % 25 == 0:
		# 	path = "./Models/Model_Epoch_" + str(i+1)	
		# 	save_path = saver.save(sess, path)

	# Close the env and write monitor result info to disk
	env.close()

		