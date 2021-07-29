#!/usr/bin/env python

from gym import Env
from gym.spaces import Discrete

import threading
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.signal
# %matplotlib inline
from helper import *
from vizdoom import *

from random import choice
from time import sleep
from time import time

import os
import cv2



# Copies one set of variables to another.
# Used to set worker network parameters to those of global network.
def update_target_graph(from_scope,to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var,to_var in zip(from_vars,to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder

# Processes Doom screen image to produce cropped and resized image. 
def process_frame(frame):
    # s = frame[10:-10,30:-30]
    s = frame
    s = scipy.misc.imresize(s,[20,20,3])
    # s = scipy.misc.imresize(s,[84,84,3])
    s = np.reshape(s,[np.prod(s.shape)]) / 255.0
    return s

# Discounting function used to calculate discounted returns.
def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

#Used to initialize weights for policy and value output layers
def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer




'''
###################################################
################# Environment #####################
###################################################
'''
class Environment(Env):
    def __init__(self, size, closed):
        self.action_space = Discrete(3)
        
        self.size = size
        
        if(closed):
            self.Map = np.zeros((size))
            self.Map[0,:] = np.ones((size[1]))
            self.Map[:,0] = np.ones((size[0]))
            self.Map[-1,:] = np.ones((size[1]))
            self.Map[:,-1] = np.ones((size[0]))
        else:
            self.Map = np.ones((size))
            
        self.state = np.array((random.randint(1,size[0]-3), random.randint(1,size[1]-3)))

        self.goal_vec = np.random.randint(1,size[0]-2,(5,2))

        self.Map[(self.state[0],self.state[1])] = 5
        self.Map[(self.goal_vec[0][0],self.goal_vec[0][1])] = 3
        # for i in range(5):
            # self.Map[(self.goal_vec[i][0],self.goal_vec[i][1])] = 3

        self.distance = np.sqrt( (self.goal_vec[0][0]-self.state[0])**2 + (self.goal_vec[0][0]-self.state[1])**2 ).astype(float)
        
        # self.observation_space = np.array([self.create_image()])
        self.observation_space = self.create_image()
        


    def create_image(self):
        m,n = self.Map.shape

        image = np.zeros((m,n,3))
        for i in range(m):
            for j in range(n):
                if(self.Map[i,j] == 0):
                    image[i,j] = [255,255,255]
                elif(self.Map[i,j] == 3):
                    image[i,j] = [0,0,255]
                elif(self.Map[i,j] == 5):
                    image[i,j] = [0,255,0]


        return image


        
                    
    def reset(self):
        idx = np.where(self.Map == 5)
        self.Map[idx] = 0
        idx = np.where(self.Map == 3)
        self.Map[idx] = 0

        size = self.Map.shape

        self.state = np.array((random.randint(1,size[0]-3), random.randint(1,size[1]-3)))
        
        self.goal_vec = np.random.randint(1,size[0]-2,(5,2))

        self.Map[(self.state[0],self.state[1])] = 5

        self.Map[(self.goal_vec[0][0],self.goal_vec[0][1])] = 3
        # for i in range(5):
        #     self.Map[(self.goal_vec[i][0],self.goal_vec[i][1])] = 3

        # self.observation_space = np.array([self.create_image()])

        self.distance = np.sqrt( (self.goal_vec[0][0]-self.state[0])**2 + (self.goal_vec[0][0]-self.state[1])**2 ).astype(float)

        self.observation_space = self.create_image()

        return self.observation_space,self.state,self.goal_vec
            
    
    def step(self, action,goal):
        flag_reward = False
        states_before = self.state
        if(action == 0):
            if(self.state[0] > 0):
                flag_reward = True
                self.state[0] = self.state[0]-1
        elif(action == 1):
            if(self.state[1] < self.size[1] - 1):
                flag_reward = True
                self.state[1] = self.state[1]+1
        elif(action == 2):
            if(self.state[0] < self.size[0] - 1):
                flag_reward = True
                self.state[0] = self.state[0]+1
        elif(action == 3):
            if(self.state[1] > 0):
                flag_reward = True
                self.state[1] = self.state[1]-1


        if(self.state[0] == goal[0] and self.state[1] == goal[1]):
            done = True
            reward = 10.0
            self.observation_space[self.state[0],self.state[1]] = [0,255,0]
        else:
            done = False

            if(self.Map[states_before[0],states_before[1]] == 1):
                self.observation_space[states_before[0],states_before[1]] = [0,0,0]
            else:
                self.observation_space[states_before[0],states_before[1]] = [255,255,255]

            if(flag_reward == True):
                reward = self.compute_reward(self.state,goal)
            else:
                reward = -1.0
            

        return self.observation_space, reward, done
        # return [np.array([self.state]), self.observation_space], reward, done
    
    def isobstacle(self, state):
        # if((self.observation_space[0,state[0],state[1]] == [0,0,0]).all()):
        if((self.observation_space[state[0],state[1]] == [0,0,0]).all()):
        #if(self.observation_space[state[0],state[1]] == 1):
            return 1
        # elif((self.observation_space[0,state[0],state[1]] == [0,0,255]).all()):
        elif((self.observation_space[state[0],state[1]] == [0,0,255]).all()):
            return 10
        # elif((self.observation_space[0,state[0],state[1]] == [0,255,0]).all()):
        elif((self.observation_space[state[0],state[1]] == [0,255,0]).all()):
            return -0.2
        else:
            return -0.6
    
    def compute_reward(self,state, goal):
        
        D = np.sqrt( (goal[0]-state[0])**2 + (goal[1]-state[1])**2 ).astype(float)

        size = self.Map.shape

        re = (self.distance - D)*self.isobstacle(state)
        self.distance = D

        # self.observation_space[0,state[0],state[1]] = [0,255,0]
        self.observation_space[state[0],state[1]] = [0,255,0]
                
        return re


'''
###################################################
################# AC Network ######################
###################################################
'''
class AC_Network():
    def __init__(self,pic_size,s_size,a_size,scope,trainer):
        with tf.variable_scope(scope):
            #Input and visual encoding layers
            # self.inputs = tf.placeholder(shape=[None,s_size],dtype=tf.float32)
            # self.imageIn = tf.reshape(self.inputs,shape=[-1,84,84,1])
            # self.conv1 = slim.conv2d(activation_fn=tf.nn.elu,
            #     inputs=self.imageIn,num_outputs=16,
            #     kernel_size=[8,8],stride=[4,4],padding='VALID')
            # self.conv2 = slim.conv2d(activation_fn=tf.nn.elu,
            #     inputs=self.conv1,num_outputs=32,
            #     kernel_size=[4,4],stride=[2,2],padding='VALID')
            # hidden = slim.fully_connected(slim.flatten(self.conv2),256,activation_fn=tf.nn.elu)
            self.inputs = tf.placeholder(shape=[None,s_size],dtype=tf.float32)
            self.imageIn = tf.reshape(self.inputs,shape=[-1,pic_size[0],pic_size[1],pic_size[2]])
            # self.imageIn = tf.reshape(self.inputs,shape=[-1,84,84,1])
            self.conv1 = slim.conv2d(activation_fn=tf.nn.elu,
                inputs=self.imageIn,num_outputs=16,
                kernel_size=[8,8],stride=[4,4],padding='VALID')
            self.conv2 = slim.conv2d(activation_fn=tf.nn.elu,
                inputs=self.conv1,num_outputs=32,
                kernel_size=[4,4],stride=[2,2],padding='VALID')
            hidden = slim.fully_connected(slim.flatten(self.conv2),256,activation_fn=tf.nn.elu)
            
            #Recurrent network for temporal dependencies
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(256,state_is_tuple=True)
            c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
            h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
            self.state_init = [c_init, h_init]
            c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
            h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
            self.state_in = (c_in, h_in)
            rnn_in = tf.expand_dims(hidden, [0])
            step_size = tf.shape(self.imageIn)[:1]
            state_in = tf.contrib.rnn.LSTMStateTuple(c_in, h_in)
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
                lstm_cell, rnn_in, initial_state=state_in, sequence_length=step_size,
                time_major=False)
            lstm_c, lstm_h = lstm_state
            self.state_out = (lstm_c[:1, :], lstm_h[:1, :])
            rnn_out = tf.reshape(lstm_outputs, [-1, 256])
            
            #Output layers for policy and value estimations
            self.policy = slim.fully_connected(rnn_out,a_size,
                activation_fn=tf.nn.softmax,
                weights_initializer=normalized_columns_initializer(0.01),
                biases_initializer=None)
            self.value = slim.fully_connected(rnn_out,1,
                activation_fn=None,
                weights_initializer=normalized_columns_initializer(1.0),
                biases_initializer=None)
            
            #Only the worker network need ops for loss functions and gradient updating.
            if scope != 'global':
                self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
                self.actions_onehot = tf.one_hot(self.actions,a_size,dtype=tf.float32)
                self.target_v = tf.placeholder(shape=[None],dtype=tf.float32)
                self.advantages = tf.placeholder(shape=[None],dtype=tf.float32)

                self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot, [1])

                #Loss functions
                self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value,[-1])))
                self.entropy = - tf.reduce_sum(self.policy * tf.log(self.policy))
                self.policy_loss = -tf.reduce_sum(tf.log(self.responsible_outputs)*self.advantages)
                self.loss = 0.5 * self.value_loss + self.policy_loss - self.entropy * 0.01

                #Get gradients from local network using local losses
                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.loss,local_vars)
                self.var_norms = tf.global_norm(local_vars)
                grads,self.grad_norms = tf.clip_by_global_norm(self.gradients,40.0)
                
                #Apply local gradients to global network
                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.apply_grads = trainer.apply_gradients(zip(grads,global_vars))





'''
###################################################
################# Class Worker ####################
###################################################
'''
class Worker():
    def __init__(self,game,name,pic_size,s_size,a_size,trainer,model_path,global_episodes):
        self.name = "worker_" + str(name)
        self.number = name        
        self.model_path = model_path
        self.trainer = trainer
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []
        self.summary_writer = tf.summary.FileWriter("train_"+str(self.number))

        #Create the local copy of the network and the tensorflow op to copy global paramters to local network
        self.local_AC = AC_Network(pic_size,s_size,a_size,self.name,trainer)
        self.update_local_ops = update_target_graph('global',self.name)        
        
        self.actions = self.actions = np.identity(a_size,dtype=bool).tolist()
        self.env = game
        
    def train(self,rollout,sess,gamma,bootstrap_value):
        rollout = np.array(rollout)
        observations = rollout[:,0]
        actions = rollout[:,1]
        rewards = rollout[:,2]
        next_observations = rollout[:,3]
        values = rollout[:,5]
        
        # Here we take the rewards and values from the rollout, and use them to 
        # generate the advantage and discounted returns. 
        # The advantage function uses "Generalized Advantage Estimation"
        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = discount(self.rewards_plus,gamma)[:-1]
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = rewards + gamma * self.value_plus[1:] - self.value_plus[:-1]
        advantages = discount(advantages,gamma)

        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        feed_dict = {self.local_AC.target_v:discounted_rewards,
            self.local_AC.inputs:np.vstack(observations),
            self.local_AC.actions:actions,
            self.local_AC.advantages:advantages,
            self.local_AC.state_in[0]:self.batch_rnn_state[0],
            self.local_AC.state_in[1]:self.batch_rnn_state[1]}
        v_l,p_l,e_l,g_n,v_n, self.batch_rnn_state,_ = sess.run([self.local_AC.value_loss,
            self.local_AC.policy_loss,
            self.local_AC.entropy,
            self.local_AC.grad_norms,
            self.local_AC.var_norms,
            self.local_AC.state_out,
            self.local_AC.apply_grads],
            feed_dict=feed_dict)
        return v_l / len(rollout),p_l / len(rollout),e_l / len(rollout), g_n,v_n
        
    def work(self,max_episode_length,gamma,sess,coord,saver):
        episode_count = sess.run(self.global_episodes)
        total_steps = 0
        print ("Starting worker " + str(self.number))
        count_k = cont_e = 0
        with sess.as_default(), sess.graph.as_default():                 
            while not coord.should_stop():
                sess.run(self.update_local_ops)
                episode_buffer = []
                episode_values = []
                episode_frames = []
                episode_reward = 0
                episode_step_count = 0
                d = False
                
                # reset
                # self.env.new_episode()
                s,robot_state,goalvec = self.env.reset()
                goal = goalvec[0]
                # get inputs
                # s = self.env.get_state().screen_buffer
                episode_frames.append(s)
                s = process_frame(s)
                rnn_state = self.local_AC.state_init
                self.batch_rnn_state = rnn_state
                flag_done = False
                # while self.env.is_episode_finished() == False:
                if self.name == 'worker_0':
                    print("Episode: %d" % count_k)
                    count_k += 1
                while flag_done == False:
                    # print("Epochs: %d" % cont_e)
                    cont_e += 1

                    #Take an action using probabilities from policy network output.
                    a_dist,v,rnn_state = sess.run([self.local_AC.policy,self.local_AC.value,self.local_AC.state_out], 
                        feed_dict={self.local_AC.inputs:[s],
                        self.local_AC.state_in[0]:rnn_state[0],
                        self.local_AC.state_in[1]:rnn_state[1]})
                    # Choose action
                    a = np.random.choice(a_dist[0],p=a_dist[0])
                    a = np.argmax(a_dist == a)

                    # STEP, r = reward (-0.01,-0.06,1.0)
                    # r = self.env.make_action(self.actions[a]) / 100.0
                    action = np.where(np.array(self.actions[a]).astype(int)==1)[0][0]
                    # print("action = %d" % action)
                    new_state, reward, flag_done  = self.env.step(action, goal)
                    r = reward/100.0
                    # print("reward = %f\n" % r)
                    # Check Done
                    # d = self.env.is_episode_finished()
                    d = flag_done
                    if d == False:
                        # s1 = self.env.get_state().screen_buffer
                        s1 = new_state
                        episode_frames.append(s1)
                        s1 = process_frame(s1)
                    else:
                        # cv2.imshow('image',new_state)
                        # cv2.waitKey(0)
                        s1 = s
                        
                    episode_buffer.append([s,a,r,s1,d,v[0,0]])
                    episode_values.append(v[0,0])

                    episode_reward += r
                    s = s1                    
                    total_steps += 1
                    episode_step_count += 1
                    
                    # # If the episode hasn't ended, but the experience buffer is full, then we
                    # # make an update step using that experience rollout.
                    if len(episode_buffer) == 30 and d != True and episode_step_count != max_episode_length - 1:
                        # Since we don't know what the true final return is, we "bootstrap" from our current
                        # value estimation.
                        v1 = sess.run(self.local_AC.value, 
                            feed_dict={self.local_AC.inputs:[s],
                            self.local_AC.state_in[0]:rnn_state[0],
                            self.local_AC.state_in[1]:rnn_state[1]})[0,0]
                        v_l,p_l,e_l,g_n,v_n = self.train(episode_buffer,sess,gamma,v1)
                        episode_buffer = []
                        sess.run(self.update_local_ops)
                    if d == True:
                        break
                                            
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_step_count)
                self.episode_mean_values.append(np.mean(episode_values))

                # print("episode_reward: %f\n" % episode_reward)
                
                # Update the network using the episode buffer at the end of the episode.
                if len(episode_buffer) != 0:
                    v_l,p_l,e_l,g_n,v_n = self.train(episode_buffer,sess,gamma,0.0)
                                
                    
                # Periodically save gifs of episodes, model parameters, and summary statistics.
                if episode_count % 5 == 0 and episode_count != 0:
                    if self.name == 'worker_0' and episode_count % 25 == 0:
                        time_per_step = 0.05
                        # time_per_step = 1.0
                        images = np.array(episode_frames)
                        make_gif(images,'./frames/image'+str(episode_count)+'.gif',
                            duration=len(images)*time_per_step,true_image=True,salience=False)
                    if episode_count % 250 == 0 and self.name == 'worker_0':
                        saver.save(sess,self.model_path+'/model-'+str(episode_count)+'.cptk')
                        print ("Saved Model")

                    mean_reward = np.mean(self.episode_rewards[-5:])
                    mean_length = np.mean(self.episode_lengths[-5:])
                    mean_value = np.mean(self.episode_mean_values[-5:])
                    summary = tf.Summary()
                    summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
                    summary.value.add(tag='Perf/Length', simple_value=float(mean_length))
                    summary.value.add(tag='Perf/Value', simple_value=float(mean_value))
                    summary.value.add(tag='Losses/Value Loss', simple_value=float(v_l))
                    summary.value.add(tag='Losses/Policy Loss', simple_value=float(p_l))
                    summary.value.add(tag='Losses/Entropy', simple_value=float(e_l))
                    summary.value.add(tag='Losses/Grad Norm', simple_value=float(g_n))
                    summary.value.add(tag='Losses/Var Norm', simple_value=float(v_n))
                    self.summary_writer.add_summary(summary, episode_count)

                    self.summary_writer.flush()
                if self.name == 'worker_0':
                    sess.run(self.increment)
                episode_count += 1







'''
###################################################
################# MAIN #####################
###################################################
'''

max_episode_length = 300
gamma = .99 # discount rate for advantage estimation and reward discounting
# s_size = 7056 # Observations are greyscale frames of 84 * 84 * 1
s_size = 1200 # Observations are rgb frames of 20 * 20 * 3
# s_size = 21168 # Observations are greyscale frames of 84 * 84 * 1
a_size = 4 # Agent can move Left, Right, Up, Down
load_model = False
model_path = './model'


## Create environmento
size_env = (20,20)
env = Environment(size_env,closed = True)
####

pic_size = env.observation_space.shape


tf.reset_default_graph()

if not os.path.exists(model_path):
     os.makedirs(model_path)
    
#Create a directory to save episode playback gifs to
if not os.path.exists('./frames'):
     os.makedirs('./frames')

# with tf.device("/cpu:0"): 
global_episodes = tf.Variable(0,dtype=tf.int32,name='global_episodes',trainable=False)
trainer = tf.train.AdamOptimizer(learning_rate=1e-4)

# Generate global network
master_network = AC_Network(pic_size,s_size,a_size,'global',None) # Generate global network
num_workers = multiprocessing.cpu_count() # Set workers to number of available CPU threads

#### Single thread
# worker = Worker(env,0,pic_size,s_size,a_size,trainer,model_path,global_episodes)
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# saver = tf.train.Saver(max_to_keep=5)
# coord = tf.train.Coordinator()
# # ckpt = tf.train.get_checkpoint_state(model_path)
# # saver.restore(sess,ckpt.model_checkpoint_path)
# worker.work(max_episode_length,gamma,sess,coord,saver)




#### Multi thread
workers = []
# Create worker classes
for i in range(num_workers):
    workers.append(Worker(env,i,pic_size,s_size,a_size,trainer,model_path,global_episodes))
saver = tf.train.Saver(max_to_keep=5)

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    if load_model == True:
        print ('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver.restore(sess,ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())
        
    # This is where the asynchronous magic happens.
    # Start the "work" process for each worker in a separate threat.
    worker_threads = []
    for worker in workers:
        worker_work = lambda: worker.work(max_episode_length,gamma,sess,coord,saver)
        t = threading.Thread(target=(worker_work))
        t.start()
        sleep(0.5)
        worker_threads.append(t)
    coord.join(worker_threads)