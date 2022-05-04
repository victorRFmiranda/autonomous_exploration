import tensorflow.compat.v1 as tf
import numpy as np

from environment.environment import Environment
from environment.environment_node_data import Mode

# from queue import Queue
import action_mapper
import queue
from multiprocessing import Queue





class NetworkVP:
	def __init__(self, device, model_name, num_actions, load = False):
		self.device = device
		self.model_name = model_name
		self.num_actions = num_actions

		self.observation_size=684+128
		# self.observation_size=1081+128
		self.observation_channels = 4
		self.batch_size = 128

		self.prediction_q = Queue(maxsize=100)

		self.learning_rate = 0.0003
		self.beta = 0.01
		self.log_epsilon = 1e-6

		self.load_check = load

		self.graph = tf.Graph()
		with self.graph.as_default() as g:
			with tf.device(self.device):
				self._create_graph()

				self.sess = tf.Session(
					graph=self.graph,
					config=tf.ConfigProto(
						allow_soft_placement=True,
						log_device_placement=False,
						gpu_options=tf.GPUOptions(allow_growth=True)))
				self.sess.run(tf.global_variables_initializer())

				if self.load_check:
					vars = tf.global_variables()
					self.saver = tf.train.Saver({var.name: var for var in vars}, max_to_keep=0)
				

	def _create_graph(self):
		self.x = tf.placeholder(
			tf.float32, [None, self.observation_size, self.observation_channels], name='X')
		self.y_r = tf.placeholder(tf.float32, [None], name='Yr')

		self.var_beta = tf.placeholder(tf.float32, name='beta', shape=[])
		self.var_learning_rate = tf.placeholder(tf.float32, name='lr', shape=[])

		self.global_step = tf.Variable(0, trainable=False, name='step')

		# As implemented in A3C paper
		self.n1 = self.conv1d_layer(self.x, 9, 16, 'conv11', stride=5)
		self.n2 = self.conv1d_layer(self.n1, 5, 32, 'conv12', stride=3)
		self.action_index = tf.placeholder(tf.float32, [None, self.num_actions])
		_input = self.n2

		flatten_input_shape = _input.get_shape()
		nb_elements = flatten_input_shape[1] * flatten_input_shape[2]


		# self.flat = tf.reshape(_input, shape=[-1, nb_elements._value])
		self.flat = tf.reshape(_input, shape=[-1, nb_elements])
		self.d1 = self.dense_layer(self.flat, 256, 'dense1')

		self.logits_v = tf.squeeze(self.dense_layer(self.d1, 1, 'logits_v', func=None), axis=[1])
		self.cost_v = 0.5 * tf.reduce_sum(tf.square(self.y_r - self.logits_v), axis=0)


		self.logits_p = self.dense_layer(self.d1, self.num_actions, 'logits_p', func=None)


		self.softmax_p = (tf.nn.softmax(self.logits_p) + 0.0) / (1.0 + 0.0 * self.num_actions)
		self.selected_action_prob = tf.reduce_sum(self.softmax_p * self.action_index, axis=1)

		self.cost_p_1 = tf.log(tf.maximum(self.selected_action_prob, self.log_epsilon)) \
					* (self.y_r - tf.stop_gradient(self.logits_v))
		self.cost_p_2 = -1 * self.var_beta * \
					tf.reduce_sum(tf.log(tf.maximum(self.softmax_p, self.log_epsilon)) *
								  self.softmax_p, axis=1)
		
		self.cost_p_1_agg = tf.reduce_sum(self.cost_p_1, axis=0)
		self.cost_p_2_agg = tf.reduce_sum(self.cost_p_2, axis=0)
		self.cost_p = -(self.cost_p_1_agg + self.cost_p_2_agg)
		

		self.cost_all = self.cost_p + self.cost_v
		self.opt = tf.train.RMSPropOptimizer(
			learning_rate=self.var_learning_rate,
			decay=0.99,
			momentum=0.0,
			epsilon=0.1)


		self.train_op = self.opt.minimize(self.cost_all, global_step=self.global_step)


	def _create_tensor_board(self):
		summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
		summaries.append(tf.summary.scalar("Pcost_advantage", self.cost_p_1_agg))
		summaries.append(tf.summary.scalar("Pcost_entropy", self.cost_p_2_agg))
		summaries.append(tf.summary.scalar("Pcost", self.cost_p))
		summaries.append(tf.summary.scalar("Vcost", self.cost_v))
		summaries.append(tf.summary.scalar("LearningRate", self.var_learning_rate))
		summaries.append(tf.summary.scalar("Beta", self.var_beta))
		for var in tf.trainable_variables():
			summaries.append(tf.summary.histogram("weights_%s" % var.name, var))

		summaries.append(tf.summary.histogram("activation_n1", self.n1))
		summaries.append(tf.summary.histogram("activation_n2", self.n2))
		summaries.append(tf.summary.histogram("activation_d2", self.d1))
		summaries.append(tf.summary.histogram("activation_v", self.logits_v))
		summaries.append(tf.summary.histogram("activation_p", self.softmax_p))

		self.summary_op = tf.summary.merge(summaries)
		self.log_writer = tf.summary.FileWriter("logs/%s" % self.model_name, self.sess.graph)

	def dense_layer(self, input, out_dim, name, func=tf.nn.relu):
		in_dim = input.get_shape().as_list()[-1]
		d = 1.0 / np.sqrt(in_dim)
		with tf.variable_scope(name):
			w_init = tf.random_uniform_initializer(-d, d)
			b_init = tf.random_uniform_initializer(-d, d)
			w = tf.get_variable('w', dtype=tf.float32, shape=[in_dim, out_dim], initializer=w_init)
			b = tf.get_variable('b', shape=[out_dim], initializer=b_init)

			output = tf.matmul(input, w) + b
			if func is not None:
				output = func(output)

		return output

	def conv2d_layer(self, input, filter_size, out_dim, name, strides, func=tf.nn.relu):
		in_dim = input.get_shape().as_list()[-1]
		d = 1.0 / np.sqrt(filter_size  * in_dim)
		with tf.variable_scope(name):
			w_init = tf.random_uniform_initializer(-d, d)
			b_init = tf.random_uniform_initializer(-d, d)
			w = tf.get_variable('w',
								shape=[filter_size, filter_size, in_dim, out_dim],
								dtype=tf.float32,
								initializer=w_init)
			b = tf.get_variable('b', shape=[out_dim], initializer=b_init)

			output = tf.nn.conv2d(input, w, strides=strides, padding='SAME') + b
			if func is not None:
				output = func(output)

		return output

	def conv1d_layer(self, input, filter_size, out_dim, name, stride, func=tf.nn.relu):
		in_dim = input.get_shape().as_list()[-1]
		d = 1.0 / np.sqrt(filter_size  * in_dim)
		with tf.variable_scope(name):
			w_init = tf.random_uniform_initializer(-d, d)
			b_init = tf.random_uniform_initializer(-d, d)
			w = tf.get_variable('w',
								shape=[filter_size, in_dim, out_dim],
								dtype=tf.float32,
								initializer=w_init)
			b = tf.get_variable('b', shape=[out_dim], initializer=b_init)

			output = tf.nn.conv1d(input, w, stride=stride, padding='SAME') + b
			if func is not None:
				output = func(output)

		return output

	def __get_base_feed_dict(self):
		return {self.var_beta: self.beta, self.var_learning_rate: self.learning_rate}

	def get_global_step(self):
		step = self.sess.run(self.global_step)
		return step

	def predict_single(self, x):
		return self.predict_p(x[None, :])[0]

	def predict_v(self, x):
		prediction = self.sess.run(self.logits_v, feed_dict={self.x: x})
		return prediction

	def predict_p(self, x):
		prediction = self.sess.run(self.softmax_p, feed_dict={self.x: x})
		return prediction
	
	def predict_p_and_v(self, x):
		return self.sess.run([self.softmax_p, self.logits_v], feed_dict={self.x: x})
	
	def train(self, x, y_r, a, trainer_id):
		feed_dict = self.__get_base_feed_dict()
		feed_dict.update({self.x: x, self.y_r: y_r, self.action_index: a})
		self.sess.run(self.train_op, feed_dict=feed_dict)

	def log(self, x, y_r, a):
		feed_dict = self.__get_base_feed_dict()
		feed_dict.update({self.x: x, self.y_r: y_r, self.action_index: a})
		step, summary = self.sess.run([self.global_step, self.summary_op], feed_dict=feed_dict)
		self.log_writer.add_summary(summary, step)

	def _checkpoint_filename(self, episode):
		return 'checkpoints/%s_%08d' % (self.model_name, episode)
	
	def _get_episode_from_filename(self, filename):
		# TODO: hacky way of getting the episode. ideally episode should be stored as a TF variable
		return int(re.split('/|_|\.', filename)[2])

	def save(self, episode):
		self.saver.save(self.sess, self._checkpoint_filename(episode))

	def load(self):
		#filename = tf.train.latest_checkpoint(os.path.dirname(self._checkpoint_filename(episode=0)))
		filename = tf.train.latest_checkpoint(os.path.dirname(self._checkpoint_filename(episode=0)))
		# if Config.LOAD_EPISODE > 0:
		#	 filename = self._checkpoint_filename(Config.LOAD_EPISODE)
		self.saver.restore(self.sess, filename)
		return self._get_episode_from_filename(filename)
	   
	def get_variables_names(self):
		return [var.name for var in self.graph.get_collection('trainable_variables')]

	def get_variable_value(self, name):
		return self.sess.run(self.graph.get_tensor_by_name(name))


	def select_action(self, state):
		# PARTE 1
		self.prediction_q.put(state)
		states_ = np.zeros(
			(self.batch_size, self.observation_size, self.observation_channels),
			dtype=np.float32)
		states_[0] = self.prediction_q.get()
		size = 1
		while size < self.batch_size and not self.prediction_q.empty():
			states_[size] = self.prediction_q.get()
			size += 1
		batch = states_[:size]

		# PARTE 2
		p, v = self.predict_p_and_v(batch)

		action = np.argmax(p)
		return action





#######################################################################
######################################################################
########################################################################



env = Environment("./environment/world/room")
env.set_mode(Mode.ALL_RANDOM, False)
env.use_ditance_angle_to_end(True)
env.set_observation_rotation_size(128)
env.use_observation_rotation_size(True)
env.set_cluster_size(1)

observation, _, flag_colide, _ = env.reset()

 
queue_state = queue.Queue(maxsize = 4)


MAX_EPISODES = 10

MAX_STEPS = 300


# def select_action(sess,state):
# 	global logits_p, logits_v, X
# 	prediction, value = sess.run([logits_p, logits_v], feed_dict={X: state})
# 	action = np.argmax(prediction)
# 	return action

# def select_action(state):
# 	p, v = self.server.model.predict_p_and_v(batch)

def update_frame(q, frame):
	if q.full():
		q.get()
	q.put(frame)

# graph = tf.get_default_graph()

# with tf.Session() as sess:
# 	new_saver = tf.train.import_meta_graph('network_00090000.meta')
# 	new_saver.restore(sess, tf.train.latest_checkpoint('./'))
# 	# weight = sess.run(['beta:0'])
# 	# print(weight)

# 	logits_p = graph.get_tensor_by_name("logits_p/b:0")
# 	logits_v = graph.get_tensor_by_name("logits_v/b:0")
# 	X = graph.get_tensor_by_name("X:0")


MODEL = NetworkVP('gpu:0', 'network', action_mapper.ACTION_SIZE)

num_actions = 7

ep = 0
while (ep < MAX_EPISODES):
	queue_state.queue.clear()
	observation, _, flag_colide, _ = env.reset()

	while(not queue_state.full()):
		update_frame(queue_state, observation)
		observation, _, flag_colide, _ = env.step(0.0,0.0,20)

	if(flag_colide):
		flag_colide = False
		continue

	state = np.array(queue_state.queue)
	state = np.transpose(state, [1, 0])  # move channels
	state = np.expand_dims(state, axis=0)

	for step in range(MAX_STEPS):

		# prediction_q.put(state)

		# state_test = np.zeros((1,1209,4))
		# prediction, value = sess.run([logits_p, logits_v], feed_dict={X: state_test})
		# action = select_action(sess,state)
		action = MODEL.select_action(state)
		print(action)
		# input("WAIT")

		linear, angular = action_mapper.map_action(action)
		next_observation, reward, done, _ = env.step(linear, angular, 20)
		update_frame(queue_state, next_observation)
		next_state = np.array(queue_state.queue)
		next_state = np.transpose(next_state, [1, 0])
		state = np.expand_dims(next_state, axis=0)

		env.visualize()


	ep += 1












# p, v = predict(sess, state_test)

# print(p)


# ## EPISODE
# def run_episode():
#	 self.env.reset()
#	 done = False
#	 experiences = []

#	 time_count = 0
#	 reward_sum = 0.0

#	 step_iteration = 0

#	 while not done:
#		 # very first few frames
#		 if self.env.current_state is None:
#			 self.env.step(None)  # 0 == NOOP
#			 continue

#		 prediction, value = predict(sess, state)
#		 action = action = np.argmax(prediction)
#		 # STEP
#		 reward, done = self.env.step(action)
#		 reward_sum += reward

#		 if Config.MAX_STEP_ITERATION < step_iteration:
#			 step_iteration = 0
#			 done = True
#			 break

#		 step_iteration += 1
#		 time_count += 1