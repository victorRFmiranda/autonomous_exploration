#!/usr/bin/env python

import rospy
import numpy as np

import gym
import time
import copy
import os
import cv2

from tf.transformations import quaternion_from_euler, euler_from_quaternion
from geometry_msgs.msg import Pose, PoseStamped, Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from nav_msgs.msg import Odometry, OccupancyGrid
from autonomous_exploration.msg import frontier

from math import pi, atan2, tan, cos, sin, sqrt, hypot, floor, ceil, log
from a_star import Astar, control


class StageEnvironment(gym.Env):
	def __init__(self,args):
		self.action_space = args.num_actions		# num frontiers (fixed)
		# self.observation_space = args.num_states
		self.observation_space = np.asarray([])
		self.max_actions = args.MAX_STEPS			# num actions for epoch (how many times check all frontiers)
		self.num_initstates = args.NUM_EPISODES 	# num start positions before change the map
		self.maps = args.maps_gt					# vector with name of stage maps for training
		self.map_count = 0

		self.init_pose = [-20.0, -20.0, 0.0]		# x, y, theta  -- Came from a parameters ?
		self.robot_pose = [0.0, 0.0, 0.0]			# x, y, theta
		self.f_points = []
		self.step_count = 0
		self.map = np.asarray([])
		self.frontier = np.asarray([])

		self.resol = 0
		self.width = 0
		self.height = 0
		self.origem_map = [0,0]
		self.size = [0,0]
		self.ocupation_map = []
		self.controlador = control()


		os.system("gnome-terminal -- roslaunch autonomous_exploration test_stage.launch map:="+self.maps[self.map_count])
		rospy.sleep(1)
		os.system("gnome-terminal -- roslaunch autonomous_exploration gmapping.launch xmin:=-25.0 ymin:=-25.0 xmax:=25.0 ymax:=25.0 delta:=0.5 odom_fram:=world")
		rospy.sleep(1)


		rospy.init_node("Stage_environment", anonymous=True)
		rospy.Subscriber("/base_pose_ground_truth", Odometry, self.callback_pose)
		rospy.Subscriber("/frontier_points", frontier, self.callback_frontier)
		rospy.Subscriber("/map_image", Image, self.callback_image)
		rospy.Subscriber("/map",OccupancyGrid,self.callback_map)
		self.pub_pose = rospy.Publisher("/cmd_pose", Pose, queue_size=1)
		self.pub_vel = rospy.Publisher("/cmd_vel", Twist, queue_size=1)


	# change the map for training
	def reset_map(self):
		node = "/stageros"
		os.system("rosnode kill "+ node)
		time.sleep(1)
		os.system("gnome-terminal -x roslaunch autonomous_exploration test_stage.launch map:="+self.maps[self.map_count])
		time.sleep(1)
		print("map changed")
		self.map_count += 1

	# restart stage (robot back to the init position) - change the robot pose in training code
	def reset_pose(self, data):
		msg_pos = Pose()
		q = quaternion_from_euler(0,0,data[2])
		msg_pos.orientation.x = q[0]
		msg_pos.orientation.y = q[1]
		msg_pos.orientation.z = q[2]
		msg_pos.orientation.w = q[3]
		msg_pos.position.x = data[0]
		msg_pos.position.y = data[1]
		msg_pos.position.z = 0.0
		self.pub_pose.publish(msg_pos)
		print("Pose reseted\n")

		new_state = np.asarray([self.robot_pose, self.frontier, self.map])

		return new_state

		# Reset Gmapping
		# node = "/GMAP"
		# os.system("rosnode kill "+ node)
		# time.sleep(1)
		# os.system("gnome-terminal -x roslaunch autonomous_exploration gmapping.launch xmin:=-25.0 ymin:=-25.0 xmax:=25.0 ymax:=25.0 delta:=0.1 odom_fram:=world")
		# time.sleep(1)
		# print("gmapping reseted")

	def reset(self):
		rospy.wait_for_service('reset_positions')

		new_state = np.asarray([self.robot_pose, self.frontier, self.map])

		return new_state


	# move robot to the selected frontier (action)
	def follow_path(self,point):
		start = (int(round((self.robot_pose[0]-self.origem_map[0]-self.resol/2.0)/self.resol)),int(round((self.robot_pose[1]-self.origem_map[1]-self.resol/2.0)/self.resol)))
		goal = (int(round((point[0]-self.origem_map[0]-self.resol/2.0)/self.resol)),int(round((point[1]-self.origem_map[1]-self.resol/2.0)/self.resol)))
		print("Start = ", start)
		print("Goal = ", goal)
		print("computing Path")
		path = None
		while path is None:
			path = Astar(self.ocupation_map, start, goal)

		vec_path = np.zeros((len(path),2))
		for i in range(len(path)):
			s = list(path[i])
			vec_path[i,:] = list(path[i])
			vec_path[i,0] = self.origem_map[0] + (vec_path[i,0]*self.resol + self.resol/2.0)
			vec_path[i,1] = self.origem_map[1] + (vec_path[i,1]*self.resol + self.resol/2.0)

		t_x = []
		t_y = []
		t_y = vec_path[:,1]
		t_x = vec_path[:,0]

		vel_msg = Twist()
		for i in range(len(t_x)):
			D = 1000
			while(D > 0.1 and not rospy.is_shutdown()):
				# D = math.sqrt((t_y[i]-robot_states[1])**2+(t_x[i]-robot_states[0])**2)
				D = _dist([t_x[i],t_y[i]],[self.robot_pose[0],self.robot_pose[1]])

				vel_msg.linear.x, vel_msg.angular.z = self.controlador.control_([t_x[i],t_y[i]],self.robot_pose)
				self.pub_vel.publish(vel_msg)

		rospy.sleep(4)



		




	def detect_action(self,action):
		if(action ==0):
			outp = self.frontier[0]
		elif(action ==1):
			outp = self.frontier[1]
		elif(action==2):
			outp = self.frontier[2]
		else:
			outp = self.frontier[3]

		return outp

	def step(self, action, flag_change):
		f_def = self.detect_action(action)
		D = _dist(self.robot_pose,f_def)
		D_min = 1000000
		for i in range(len(self.frontier)):
			Ds = _dist(self.robot_pose,self.frontier[i])
			if(Ds < D_min):
				D_min = Ds
				best_frontier = self.frontier[i]


		# self.follow_path(f_def)
		reward = self.compute_reward(D,D_min)


		if(self.step_count >= self.max_actions):
			done = True
			self.step_count = 0
		else:
			done = False


		self.step_count += 1


		if(flag_change == True):
			self.follow_path(best_frontier)
			flag_change = False
			new_state = np.asarray([self.robot_pose, self.frontier, self.map])
			return new_state, reward, done

		else:
			new_state = np.asarray([self.robot_pose, self.frontier, self.map])

			return new_state, reward, done


	# compute the reward for this action
	def compute_reward(self,D,D_min):

		if (D <= D_min):
			re = 1
			# done = False
		else:
			re = -1
			# done = True

		return re


	def callback_pose(self, data):
		self.robot_pose[0] = data.pose.pose.position.x
		self.robot_pose[1] = data.pose.pose.position.y
		q = [data.pose.pose.orientation.x,data.pose.pose.orientation.y,data.pose.pose.orientation.z,data.pose.pose.orientation.w]
		angles = euler_from_quaternion(q)
		self.robot_pose[2] = angles[2]
		self.robot_pose = np.asarray(self.robot_pose)

	def callback_frontier(self, data):
		self.frontier = []
		for i in range(len(data.clusters)):
			self.frontier.append([data.clusters[i].x,data.clusters[i].y])

		self.frontier = np.asarray(self.frontier)

	def callback_image(self, data):
		bridge = CvBridge()
		img2 = bridge.imgmsg_to_cv2(data, desired_encoding='rgb8')
		# self.ocupation_map = img2
		img = cv2.resize(img2, (64, 64))
		self.map = img.transpose()
		# self.map = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		self.observation_space = np.array([self.robot_pose,self.frontier,self.map])

	def callback_map(self, data):
		self.resol = data.info.resolution
		self.width = data.info.width
		self.height = data.info.height
		self.origem_map = [data.info.origin.position.x,data.info.origin.position.y]
		self.size = [self.origem_map[0]+(data.info.width * self.resol),self.origem_map[1]+(data.info.height * self.resol)]

		self.ocupation_map = np.asarray(data.data, dtype=np.int8).reshape(data.info.height, data.info.width)
		self.ocupation_map = np.where(self.ocupation_map==0,0,1)





def _PotControl(robot_pose,goal):
	d = 0.2
	k = 1
	#pot field
	Ux = k * (goal[0] - robot_pose[0])
	Uy = k * (goal[1] - robot_pose[1])
	#feedback_linearization
	vx = cos(robot_pose[2]) * Ux + sin(robot_pose[2]) * Uy
	w = -(sin(robot_pose[2]) * Ux)/ d + (cos(robot_pose[2]) * Uy) / d
	return vx,w


def _dist(p1,p2): 
	return ((p1[0]-p2[0])**2 +(p1[1]-p2[1])**2)**(0.5)


'''
from config import Config
args = Config().parse()
# set argumments
# args.num_actions = 1
# args.max_action_count = 2
# args.n_init = 4
# args.maps_gt = 5
env = StageEnvironment(args)

cont = 0
rate = rospy.Rate(1)
goal = [-20,-15]
init_pose = env.init_pose


while not rospy.is_shutdown():
	if(env.observation_space.shape[0] != 0):
		print(env.observation_space.shape)

	rate.sleep()

'''