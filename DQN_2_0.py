import tensorflow as tf
import numpy as np
import random
from collections import deque
import gym
import cv2

class DQN:

	# Hyper Parameters
	ACTION = 9
	FRAME_PER_ACTION = 1
	GAMMA = 0.99 # decay rate of past observations
	OBSERVE = 100000 # time steps to observe before traing
	EXPLORE = 15000
	FINAL_EPSILON = 0.1
	INITIAL_EPSILON = 1.0
	REPLAY_MEMORY = 50000
	BATCH_SIZE = 32 # size of minibatch

	def __init__(self):
		self.replay_memory = deque()
		self.create_Q_network()
		self.time_step = 0
		self.epsilon = self.INITIAL_EPSILON

	def create_Q_network(self):
		# network weights
		W_conv1 = self.weight_variable([8, 8, 4, 32])
		b_conv1 = self.bias_variable([32])

		W_conv2 = self.weight_variable([4, 4, 32, 64])
		b_conv2 = self.bias_variable([64])

		W_conv3 = self.weight_variable([3, 3, 64, 64])
		b_conv3 = self.bias_variable([64])

		W_fc1 = self.weight_variable([6400, 512])
		b_fc1 = self.bias_variable([512])

		W_fc2 = self.weight_variable([512, self.ACTION])
		b_fc2 = self.bias_variable([self.ACTION])

		# input layer
		self.state_input = tf.placeholder("float", [None, 160, 160, 4])

		# hidden layer
		h_conv1 = tf.nn.relu(self.conv2d(self.state_input, W_conv1, 4) + b_conv1)
		h_pool1 = self.max_pool_2x2(h_conv1)

		h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2, 2) + b_conv2)

		h_conv3 = tf.nn.relu(self.conv2d(h_conv2, W_conv3, 1) + b_conv3)

		h_conv3_flat = tf.reshape(h_conv3, [-1, 6400])
		h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

		# Q Value layer
		self.Q_value = tf.matmul(h_fc1, W_fc2) + b_fc2

		self.action_input = tf.placeholder("float", [None, self.ACTION])
		self.y_input = tf.placeholder("float", [None])
		Q_action = tf.reduce_sum(tf.mul(self.Q_value, self.action_input), reduction_indices = 1)
		self.cost = tf.reduce_mean(tf.square(self.y_input - Q_action))
		self.train_step = tf.train.AdamOptimizer(1e-6).minimize(self.cost)

		# saving and loading networks
		self.saver = tf.train.Saver()
		self.session = tf.InteractiveSession()
		self.session.run(tf.initialize_all_variables())
		checkpoint = tf.train.get_checkpoint_state("saved_networks")
		if checkpoint and checkpoint.model_checkpoint_path:
			saver.restore(self.session, checkpoint.model_checkpoint_path)
			print "Successfully loaded:", checkpoint.model_checkpoint_path
		else:
			print "Could not find old network weights"

	def train_Q_Network(self):
		# Step 1: obtain random minibatch from replay memory
		minibatch = random.sample(self.replay_memory,self.BATCH_SIZE)
		state_batch = [data[0] for data in minibatch]
		action_batch = [data[1] for data in minibatch]
		reward_batch = [data[2] for data in minibatch]
		nextState_batch = [data[3] for data in minibatch]
		# Step 2: calculate y 
		y_batch = []
		Q_Value_batch = self.Q_value.eval(feed_dict={self.state_input:nextState_batch})
		for i in range(0,self.BATCH_SIZE):
			terminal = minibatch[i][4]
			if terminal:
				y_batch.append(reward_batch[i])
			else:
				y_batch.append(reward_batch[i] + self.GAMMA * np.max(Q_Value_batch[i]))


		self.train_step.run(feed_dict={
			self.y_input : y_batch,
			self.action_input : action_batch,
			self.state_input : state_batch
			})

        # save network every 10000 iteration
		#print self.time_step
		if self.time_step % 10000 == 0:
			print "save", self.time_step
			self.saver.save(self.session, 'saved-networks/' + 'dqn', global_step = self.time_step)


	def setPerception(self,nextObservation,action,reward,terminal):
		new_state = np.append(nextObservation,self.current_state[:,:,1:],axis = 2)
		self.replay_memory.append((self.current_state,action,reward,new_state,terminal))
		if len(self.replay_memory) > self.REPLAY_MEMORY:
			self.replay_memory.popleft()
		if self.time_step > self.OBSERVE:
            # Train the network
			self.train_Q_Network()

		self.current_state = new_state
		self.time_step += 1

	def getAction(self):
		Q_value = self.Q_value.eval(feed_dict ={
			self.state_input:[self.current_state]
			})[0]
		action = np.zeros(self.ACTION)
		action_index = 0
		if self.time_step % self.FRAME_PER_ACTION == 0:
			if random.random() <= self.epsilon:
				action_index = random.randrange(self.ACTION)
			else:
				action_index = np.argmax(Q_value)

		if self.epsilon > self.FINAL_EPSILON and self.time_step > self.OBSERVE:
			self.epsilon -= (self.INITIAL_EPSILON - self.FINAL_EPSILON) /  self.EXPLORE

		return action_index

	def set_init_state(self, observation):
		self.current_state = np.stack((observation, observation, observation, observation), axis = 2)

	def weight_variable(self, shape):
		initial = tf.truncated_normal(shape, stddev = 0.01)
		return tf.Variable(initial)

	def bias_variable(self, shape):
		initial = tf.constant(0.01, shape = shape)
		return tf.Variable(initial)	
	
	def conv2d(self, x, W, stride):
		return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = 'SAME')

	def max_pool_2x2(self, x):
		return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

def print_ob(ob):
	for i in range(len(ob)):
		for j in range(len(ob[i])):
			print ob[i][j]

def preprocess(observation):
	observation = observation[0:160]
	observation = cv2.cvtColor(cv2.resize(observation, (160, 160)), cv2.COLOR_BGR2GRAY)
	#ret, observation = cv2.threshold(observation, 1, 255, cv2.THRESH_BINARY)
	ob = np.reshape(observation, (160, 160, 1))
	#print_ob(ob)
	return ob

def process_action(action):
	a = np.zeros(9)
	a[action] = 1
	return a

def main():
	env = gym.make('MsPacman-v0')
	agent = DQN()
	observation0 = env.reset()
	observation0 = observation0[0:160]
	observation0 = cv2.cvtColor(cv2.resize(observation0, (160, 160)), cv2.COLOR_BGR2GRAY)
	agent.set_init_state(observation0)
    # Step 3.2: run the game
	while True:
		action = agent.getAction()
		nextObservation,reward,terminal, info = env.step(action)
		nextObservation = preprocess(nextObservation)
		agent.setPerception(nextObservation,process_action(action),reward,terminal)
	'''
	for episode in xrange(EPISODE):
		# initialiaze task
		state = env.reset()
		#Train
		sum_reward = 0
		for step in xrange(STEP):
			action = agent.getAction(preprocess(state)) # e-greedy action for train
			next_state, reward, done, info = env.step(action)
			# Define reward for agent
			agent.perceive(preprocess(next_state), action, reward, done)
			if done:
				break
		# Test every 100 episodes
		
		if episode % 100 == 0:
			total_reward = 0
			for i in xrange(TEST):
				state = env.reset()	
				print 'i:', i
				for j in xrange(STEP):
					print 'j:', j
					env.render()
					state_flat = flatten(state.tolist())
					action = agent.action(state_flat) # direct action for test
					state, reward, done, _ = env.step(action)
					total_reward += reward
					if done:
						break
			ave_reward = total_reward / TEST
			print 'episode', episode, 'Evaluation Average Reward:', ave_reward
			if ave_reward >= 200:
				break		
		'''
main()