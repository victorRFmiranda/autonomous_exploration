#!/usr/bin/env python

import rospy
import numpy as np
import rosnode

import gym
import time
import copy
import os
import cv2

from tf.transformations import quaternion_from_euler, euler_from_quaternion
from geometry_msgs.msg import Pose, PoseStamped, Twist, Polygon, Point32, Point
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import Int32, Bool
from cv_bridge import CvBridge
from nav_msgs.msg import Odometry, OccupancyGrid
import matplotlib.pyplot as plt

from math import pi, atan2, tan, cos, sin, sqrt, hypot, floor, ceil, log

from autonomous_exploration.msg import frontier
# from a_star import Astar, control
from rrtStar import RRTStar, control, compute_obstacles
# from rrt import RRT
# from new_aStar import A_Star, control
from astar_start_end import searching_control
from dijkstra import Dijkstra
from vecfield_control.msg import Path



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
		self.frontier_anterior = np.zeros((4,2))
		self.freeMap_size = 0
		
		self.resol = 0
		self.width = 0
		self.height = 0
		self.origem_map = [0,0]
		self.size = [0,0]
		self.ocupation_map = []
		self.occ_map = OccupancyGrid()
		self.controlador = control()
		self.laser = []
		self.crash = 0
		self.flag_control = 0
		self.flag_frontier = 1


		os.system("gnome-terminal -- roslaunch autonomous_exploration test_stage.launch map:="+self.maps[self.map_count])
		rospy.sleep(1)
		os.system("gnome-terminal -- roslaunch autonomous_exploration gmapping.launch xmin:=-25.0 ymin:=-25.0 xmax:=25.0 ymax:=25.0 delta:=0.5 odom_fram:=world")
		rospy.sleep(1)


		rospy.init_node("Stage_environment", anonymous=True)
		rospy.Subscriber("/crash_stall", Int32, self.callback_crashStatus)
		rospy.Subscriber("/base_pose_ground_truth", Odometry, self.callback_pose)
		rospy.Subscriber("/frontier_points", frontier, self.callback_frontier)
		rospy.Subscriber("/map_image", Image, self.callback_image)
		rospy.Subscriber("/map",OccupancyGrid,self.callback_map)
		rospy.Subscriber('/base_scan', LaserScan, self.callback_laser)
		rospy.Subscriber("/reached_endpoint", Int32, self.callback_control)
		rospy.Subscriber("/frontier_alive", Int32, self.callback_falive)
		self.pub_pose = rospy.Publisher("/cmd_pose", Pose, queue_size=1)
		self.pub_vel = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
		self.pub_traj = rospy.Publisher("/traj_points", Path, queue_size=10)
		# self.pub_traj2 = rospy.Publisher("/traj_points", Path, queue_size=10)



	# def max_freeSpaces(self):
	# 	file = rospy.get_param("/map_dir")
	# 	img = cv2.imread(file)
	# 	freeSpace = np.sum(img == 255)
	# 	colision = np.sum(img == 0)
	# 	return freeSpace


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

		# KIll Control
		node = "/vecfield_control"
		os.system("rosnode kill "+ node)
		rospy.sleep(2)

		# Reset Gmapping
		node = "/GMAP"
		os.system("rosnode kill link1_broadcaster")
		rospy.sleep(1)
		os.system("rosnode kill Detect_frontier")
		rospy.sleep(1)
		os.system("rosnode kill "+ node)
		rospy.sleep(2)
		
		# Reset Pose
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
		rospy.sleep(3)

		# Open Gmapping
		os.system("gnome-terminal -- roslaunch autonomous_exploration gmapping.launch xmin:=-25.0 ymin:=-25.0 xmax:=25.0 ymax:=25.0 delta:=0.5 odom_fram:=world")
		rospy.sleep(1)
		print("gmapping reseted")
		rospy.sleep(10)

		while(self.flag_control == 1):
			print("wainting control")
			rospy.sleep(1)
		while(self.map.size == 0):
			print("wainting Map!")
			rospy.sleep(1)

		# new_state = np.asarray([self.robot_pose, self.frontier, self.map])
		# CHANGE HERE
		n_rpose = np.asarray([int(round((self.robot_pose[0]-self.origem_map[0]-0.32)/0.64)),int(round((self.robot_pose[1]-self.origem_map[1]-0.32)/0.64)),self.robot_pose[2]])
		n_frontier = np.zeros(self.frontier.shape)
		for k in range(len(self.frontier)):
			n_frontier[k] = [int(round((self.frontier[k][0]-self.origem_map[0]-0.32)/0.64)),int(round((self.frontier[k][1]-self.origem_map[1]-0.32)/0.64))]

		new_state = np.asarray([n_rpose, n_frontier, self.map])

		return new_state

		

	def reset(self):
		rospy.wait_for_service('reset_positions')

		# new_state = np.asarray([self.robot_pose, self.frontier, self.map])
		# CHANGE HERE
		n_rpose = np.asarray([int(round((self.robot_pose[0]-self.origem_map[0]-0.32)/0.64)),int(round((self.robot_pose[1]-self.origem_map[1]-0.32)/0.64)),self.robot_pose[2]])
		n_frontier = np.zeros(self.frontier.shape)
		for k in range(len(self.frontier)):
			n_frontier[k] = [int(round((self.frontier[k][0]-self.origem_map[0]-0.32)/0.64)),int(round((self.frontier[k][1]-self.origem_map[1]-0.32)/0.64))]

		new_state = np.asarray([n_rpose, n_frontier, self.map])

		return new_state

	
	# Dijkstra
	def follow_path(self,point):
		global flag_nodes

		start = [int(round((self.robot_pose[0]-self.origem_map[0]-self.resol/2.0)/self.resol)),int(round((self.robot_pose[1]-self.origem_map[1]-self.resol/2.0)/self.resol))]
		goal = [int(round((point[0]-self.origem_map[0]-self.resol/2.0)/self.resol)),int(round((point[1]-self.origem_map[1]-self.resol/2.0)/self.resol))]


		obst_idx = np.where(self.ocupation_map == 1)
		obstacles = [obst_idx[0].tolist(),obst_idx[1].tolist()]
		ox = obstacles[1]
		oy = obstacles[0]

		grid_size = 0.5
		robot_radius = 2.0
		dijkstra = Dijkstra(ox, oy, grid_size, robot_radius)

		try:
			# path = []

			# print("Planning")
			# rx, ry = dijkstra.planning(start[0], start[1], goal[0], goal[1])
			# # path.append(start)
			# for j in range(len(rx)):
			# 	path.append([rx[len(rx)-1-j],ry[len(rx)-1-j]])

			# # Prevent planning erors
			count_plan = 0
			path = []
			while (len(path) < 3 and count_plan < 5):
				# print("\33[91m Retry planning! \33[0m")
				print("Planning")
				path = []
				rx, ry = dijkstra.planning(start[0], start[1], goal[0], goal[1])
				# path.append(start)
				for j in range(len(rx)):
					path.append([rx[len(rx)-1-j],ry[len(rx)-1-j]])

				count_plan += 1


			if (len(path) < 3):
				planning_fail = True
				print("\33[41m Planning failure, return! \33[0m")

			else:
				planning_fail = False

				vec_path = np.zeros((len(path),2))
				for i in range(len(path)):
					# if(i==len(path)):
					# 	vec_path[i,0] = point[0]
					# 	vec_path[i,1] = point[1]
					# else:
					vec_path[i,:] = list(path[i])
					vec_path[i,0] = self.origem_map[0] + (vec_path[i,0]*self.resol + self.resol/2.0)
					vec_path[i,1] = self.origem_map[1] + (vec_path[i,1]*self.resol + self.resol/2.0)

				# spec = raw_input("Press Enter to continue")
				xy_path = create_traj_msg(vec_path)

				self.pub_traj.publish(xy_path)

				D = 1000
				count = 0
				t1 = rospy.get_rostime().to_sec()
				while(D > 0.5 and not rospy.is_shutdown()):
					if(rospy.get_rostime().to_sec() - t1 == 10):
						# print("\33[92m TIME \33[0m")
						t1 = rospy.Time.now().to_sec()
						self.pub_traj.publish(xy_path)
					D = _dist([point[0],point[1]],[self.robot_pose[0],self.robot_pose[1]])

					if not self.check_nodes():
						break

					if(self.flag_control == 1):
						self.flag_control = 0
						break

					if(self.crash):
						print("CRASH")
						break

					rospy.sleep(0.1)

				count = 0

				rospy.sleep(5)

			return planning_fail

		except:
			planning_fail = True
			return planning_fail
		






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


	def check_nodes(self):
		flag = True
		node_list = list(rosnode.get_node_names())
		# check vector field
		if not (node_list.count('/vecfield_control')):
			flag = False
			print("Reseting Control")
		# check gmapping
		if not (node_list.count('/GMAP')):
			flag = False
			print("Reseting Mapping")
		# check detect frontiers
		if not (node_list.count('/Detect_frontier')):
			flag = False
			print("Reseting Detect Frontiers")
		# check stage
		if not (node_list.count('/stageros')):
			node = "/stageros"
			os.system("rosnode kill "+ node)
			time.sleep(1)
			os.system("gnome-terminal -x roslaunch autonomous_exploration test_stage.launch map:="+self.maps[self.map_count])
			time.sleep(1)
			flag = False
			print("Reseting Stage")
		return flag


	def step(self, action):
		flag_nodes = self.check_nodes()
		# global flag_nodes

		if(self.flag_frontier and flag_nodes):

			try:
				f_def = self.detect_action(action)

				
				D = _dist(self.robot_pose,f_def)


				map_before = self.freeMap_size
				pose_before = self.robot_pose[0:2]

				# Path planning and follow
				planning_fail = self.follow_path(f_def)
				print("Control END")

				if(planning_fail):
					reward = 0

				else:

					# compute distance
					map_after = self.freeMap_size
					map_gain = map_after - map_before

					reward = self.compute_reward(D, map_gain)
			except:
				reward = 0

		else:
			reward = 0
			print("Failure!")



		if(self.step_count >= self.max_actions or reward == 0):
			done = True
			self.step_count = 0
		else:
			done = False


		self.step_count += 1


		# new_state = np.asarray([self.robot_pose, self.frontier, self.map])
		# CHANGE HERE
		n_rpose = np.asarray([int(round((self.robot_pose[0]-self.origem_map[0]-0.32)/0.64)),int(round((self.robot_pose[1]-self.origem_map[1]-0.32)/0.64)),self.robot_pose[2]])
		n_frontier = np.zeros(self.frontier.shape)
		for k in range(len(self.frontier)):
			n_frontier[k] = [int(round((self.frontier[k][0]-self.origem_map[0]-0.32)/0.64)),int(round((self.frontier[k][1]-self.origem_map[1]-0.32)/0.64))]

		new_state = np.asarray([n_rpose, n_frontier, self.map])

		return new_state, reward, done


	# compute the reward for this action
	def compute_reward(self,D, map_gain):
		# map_reward = 0.1*float(map_gain) #/float(self.freeMap_size)

		if(self.crash == 0):
			map_reward = 0.07*float(map_gain) #/float(self.freeMap_size)
			# distancy = log(D)
			distancy = 0.5*D
			
			# if (map_reward == 0):
			# 	re = 0
			# else:
			# re = 0.5*D + map_reward
			re = distancy + map_reward

			print("\33[92m Map Reward = %f \33[0m" % map_reward)
			print("\33[94m Distance Reward = %f \33[0m" % distancy)
			print("\33[96m Reward total = %f \33[0m" % re)
		else:
			re = 0

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
		self.freeMap_size = data.map_increase.data

	def callback_image(self, data):
		bridge = CvBridge()
		# img2 = bridge.imgmsg_to_cv2(data, desired_encoding='mono16')
		img2 = bridge.imgmsg_to_cv2(data, desired_encoding='rgb8')
		img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
		# np.savetxt('normal.txt', img2, delimiter=',')
		# print("\nNormal\n\n")
		# print(img2)
		img = cv2.resize(img2, (64, 64),interpolation=cv2.INTER_NEAREST)
		# print("\n\nRESIZED\n\n")
		# print(img)
		# np.savetxt('resized.txt', img2, delimiter=',')
		img = img.transpose()
		img = img/255.0
		img = img.astype('float32')
		
		self.map = np.asarray([img])

		self.observation_space = np.array([self.robot_pose,self.frontier,self.map])

	def callback_map(self, data):
		self.occ_map = data
		self.resol = data.info.resolution
		self.width = data.info.width
		self.height = data.info.height
		self.origem_map = [data.info.origin.position.x,data.info.origin.position.y]
		self.size = [self.origem_map[0]+(data.info.width * self.resol),self.origem_map[1]+(data.info.height * self.resol)]

		self.ocupation_map = np.asarray(data.data, dtype=np.int8).reshape(data.info.height, data.info.width)
		# self.ocupation_map = np.where(self.ocupation_map==0,0,1)
		self.ocupation_map = np.where(self.ocupation_map==100,1,0)


	def callback_laser(self, data):
		self.laser = data.ranges					 # Distancias detectadas
		self.l_range_max = data.range_max		  # range max do lidar
		self.l_range_min = data.range_min		  # range min do lidar

	def callback_crashStatus(self, data):
		self.crash = data.data

	def callback_control(self,data):
		self.flag_control = data.data

	def callback_falive(self, data):
		self.flag_frontier = data.data






def _dist(p1,p2): 
	return ((p1[0]-p2[0])**2 +(p1[1]-p2[1])**2)**(0.5)



def create_traj(points0,r_pos):

	N_interpolation = 5

	#Include the current robot's position in the planning
	points = [r_pos] + points0

	n = len(points)

	#Matrix associated to constrainned 3rd order polynomials
	M = [[0, 0, 0, 1],[1, 1, 1, 1],[0, 0, 1, 0],[3, 2, 1, 0]]
	Minv = [[2, -2, 1, 1],[-3, 3, -2, -1],[0, 0, 1, 0],[1, 0, 0, 0]]

	#Heurietics to define the 
	T = []
	for k in range(n):
		if k == 0:
			v = [points[k+1][0]-points[k][0], points[k+1][1]-points[k][1], points[k+1][2]-points[k][2]]
		elif k==(n-1):
			v = [points[k][0]-points[k-1][0], points[k][1]-points[k-1][1], points[k][1]-points[k-1][2]]
		else:
			v = [(points[k+1][0]-points[k-1][0])/2.0, (points[k+1][1]-points[k-1][1])/2.0, (points[k+1][1]-points[k-1][2])/2.0]
		T.append(v)

	#Definition of a vector to sample some points of the computed polynomials
	s_vec = [i/float(N_interpolation) for i in range(N_interpolation)]
	path = [[],[],[]]

	#Iterate over each pair of points
	for k in range(n-1):

		#Write the matrix that contains the boundary conditions for the polynomials
		A = [[points[k][0],points[k][1]], [points[k+1][0],points[k+1][1]], [T[k][0],T[k][1]], [T[k+1][0],T[k+1][1]]]
		#A = [[points[k][0],points[k][1],points[k][2]], [points[k+1][0],points[k+1][1],points[k+1][2]], [T[k][0],T[k][1],T[k][2]], [T[k+1][0],T[k+1][1],T[k+1][2]]]

		#Compute the coefficients
		ck = np.matrix(Minv)*np.matrix(A)
		
		#Sample the computed polynomials
		for s in s_vec:
			path[0].append(ck[0,0]*s**3+ck[1,0]*s**2+ck[2,0]*s**1+ck[3,0]*s**0)
			path[1].append(ck[0,1]*s**3+ck[1,1]*s**2+ck[2,1]*s**1+ck[3,1]*s**0)
			path[2].append(points[k][2]*(1.0-s)+points[k+1][2]*(s))

	#Include the last point on the path
	path[0].append(ck[0,0]+ck[1,0]+ck[2,0]+ck[3,0])
	path[1].append(ck[0,1]+ck[1,1]+ck[2,1]+ck[3,1])
	path[2].append(points[k+1][2])


	return path


def create_traj_msg(traj):

	# Create 'Polygon' message (array of messages of type 'Point')
	traj_msg = Path()
	p = Point()
	for k in range(len(traj[:,0])):
		# Create point
		p = Point()
		# Atribute values
		p.x = traj[k][0]
		p.y = traj[k][1]
		p.z = 0.0
		# Append point to polygon
		traj_msg.path.points.append(p)

	traj_msg.header.stamp = rospy.Time.now()

	traj_msg.closed_path_flag = False
	traj_msg.insert_n_points = 10
	traj_msg.filter_path_n_average = 4


	return traj_msg
