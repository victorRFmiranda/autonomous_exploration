#!/usr/bin/env python

import rospy
import numpy as np

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
		rospy.Subscriber("reached_endpoint", Int32, self.callback_control)
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
		rospy.sleep(5)

		new_state = np.asarray([self.robot_pose, self.frontier, self.map])

		return new_state

		

	def reset(self):
		rospy.wait_for_service('reset_positions')

		new_state = np.asarray([self.robot_pose, self.frontier, self.map])

		return new_state



	'''
	# Pot Field
	def follow_path(self,point):
		Delta = 1000
		vel_msg = Twist()
		# for f in range():
		while(Delta > 0.1 and not rospy.is_shutdown()):
			Delta = _dist([point[0],point[1]],[self.robot_pose[0],self.robot_pose[1]])
			D, alfa, obs_pos = min_dist(self.robot_pose[0],self.robot_pose[1],self.robot_pose[2],self.laser)
			alfa = float(alfa)


			U_r = pot_rep(self.robot_pose[2], D, alfa)
			U_a = pot_att(self.robot_pose[0], self.robot_pose[1], point[0], point[1])

			Ux = U_a[0] + U_r[0]
			Uy = U_a[1] + U_r[1]
			vel_msg.linear.x, vel_msg.angular.z = feedback_linearization(Ux, Uy, self.robot_pose[2])
			self.pub_vel.publish(vel_msg)

			if(self.crash):
				break

			# rospy.sleep(1)

		rospy.sleep(5)
	'''

	'''
	# RRT
	# move robot to the selected frontier (action)
	def follow_path(self,point):
		start = ((self.robot_pose[0]),(self.robot_pose[1]), (self.robot_pose[2]))
		obstacle_list = compute_obstacles(self.width,self.height,self.resol,self.origem_map,self.occ_map)
		goal = ((point[0]),(point[1]))
		rrt_path = RRTStar(start=start,goal=goal,map_size=[self.origem_map,self.size],resol_map = self.resol, occupancy_map = self.occ_map, obstacle_list=obstacle_list,max_iter=10000,step_size = 2.0, dt=0.1)
		# rrt_path = RRT(start=start,goal=goal,map_size=self.size,obstacle_list=obstacle_list,step_size=2.0,path_frac=0.2,max_iter=10000)
		print("Start = ", start)
		print("Goal = ", goal)
		print("computing Path")
		path = rrt_path.planning()
		if path is None:
			print("Path not found!")
			rrt_path.draw_graph()
			plt.show()
		else:
			new_traj = np.zeros((len(path),2))
			j = 0
			for i in range(len(path)-1,-1,-1):
				new_traj[j,0] = path[i][0]
				new_traj[j,1] = path[i][1]
				j+=1

			vel_msg = Twist()
			for i in range(len(new_traj)):
				D = 1000
				while(D > 0.1 and not rospy.is_shutdown()):
					D = _dist([new_traj[i,0],new_traj[i,1]],[self.robot_pose[0],self.robot_pose[1]])
					vel_msg.linear.x, vel_msg.angular.z = self.controlador.control_([new_traj[i,0],new_traj[i,1]],self.robot_pose)
					self.pub_vel.publish(vel_msg)

					if(self.crash):
						break

			rospy.sleep(5)
	'''
	
	
	'''
	# AStar
	def follow_path(self,point):
		start = (int(round((self.robot_pose[0]-self.origem_map[0]-self.resol/2.0)/self.resol)),int(round((self.robot_pose[1]-self.origem_map[1]-self.resol/2.0)/self.resol)))
		goal = (int(round((point[0]-self.origem_map[0]-self.resol/2.0)/self.resol)),int(round((point[1]-self.origem_map[1]-self.resol/2.0)/self.resol)))
		print("computing Path")
		path = None

		while len(self.occ_map.data)<1:
			print("Waiting Map")
			rate.sleep()
			pass

		t_x, t_y, points = A_Star(self.occ_map,start,goal)

		r_pos = [self.robot_pose[0], self.robot_pose[1], 0.0]
		# path1 = create_traj(points,r_pos)
		path = create_traj_msg(points)

		self.pub_traj.publish(path)

		D = 1000
		while(D > 0.1 and not rospy.is_shutdown()):
			D = _dist([point[0],point[1]],[self.robot_pose[0],self.robot_pose[1]])
			rospy.sleep(1)
		# while(self.flag_control >= 100):
			# rospy.sleep(1)

		rospy.sleep(10)
	'''

	'''
	# AStar 2
	def follow_path(self,point):
		start = [int(round((self.robot_pose[0]-self.origem_map[0]-self.resol/2.0)/self.resol)),int(round((self.robot_pose[1]-self.origem_map[1]-self.resol/2.0)/self.resol))]
		goal = [int(round((point[0]-self.origem_map[0]-self.resol/2.0)/self.resol)),int(round((point[1]-self.origem_map[1]-self.resol/2.0)/self.resol))]


		obst_idx = np.where(self.ocupation_map != 0)
		obstacles = [obst_idx[1].tolist(),obst_idx[0].tolist()]

		obst_list = []
		for i in range(len(obstacles[0])):
			obst_list.append([obstacles[0][i],obstacles[1][i]])

		obst_list = np.asarray(obst_list)
		# obst_list = np.asarray(obstacles)

		# print(obstacles)
		print("Computing Path")
		path = searching_control(start, goal, obst_list, obst_list)
		rospy.sleep(1)



		# Compute x,y path
		vec_path = np.zeros((len(path),2))
		for i in range(len(path)):
			s = list(path[i])
			vec_path[i,:] = list(path[i])
			vec_path[i,0] = self.origem_map[0] + (vec_path[i,0]*self.resol + self.resol/2.0)
			vec_path[i,1] = self.origem_map[1] + (vec_path[i,1]*self.resol + self.resol/2.0)

		xy_path = create_traj_msg(vec_path)

		self.pub_traj.publish(xy_path)

		D = 1000
		count = 0
		while(D > 0.6 and not rospy.is_shutdown()):
			D = _dist([point[0],point[1]],[self.robot_pose[0],self.robot_pose[1]])
			rospy.sleep(1)
			print("Distancy = %f" % D)

			if(self.flag_control >= 100):
				count += 1
				if(count >= 5):
					break

			if(self.crash):
				print("CRASH")
				break

		count = 0
		rospy.sleep(5)
	'''

	
	# Dijkstra
	def follow_path(self,point):
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
			rx, ry = dijkstra.planning(start[0], start[1], goal[0], goal[1])

			path = []
			for j in range(len(rx)):
				path.append([rx[len(rx)-1-j],ry[len(rx)-1-j]])

			vec_path = np.zeros((len(path),2))
			for i in range(len(path)):
				vec_path[i,:] = list(path[i])
				vec_path[i,0] = self.origem_map[0] + (vec_path[i,0]*self.resol + self.resol/2.0)
				vec_path[i,1] = self.origem_map[1] + (vec_path[i,1]*self.resol + self.resol/2.0)

			xy_path = create_traj_msg(vec_path)

			self.pub_traj.publish(xy_path)

			D = 1000
			count = 0
			while(D > 0.5 and not rospy.is_shutdown()):
				D = _dist([point[0],point[1]],[self.robot_pose[0],self.robot_pose[1]])
				# print("Distancy = %f" % D)

				if(self.flag_control >= 100):
					count += 1
					if(count >= 5):
						break

				if(self.crash):
					print("CRASH")
					break

				rospy.sleep(1)

			count = 0

			rospy.sleep(5)

		except:
			return
		






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


	def step(self, action):
		f_def = self.detect_action(action)

		D = _dist(self.robot_pose,f_def)


		map_before = self.freeMap_size
		pose_before = self.robot_pose[0:2]

		# Path planning and follow
		self.follow_path(f_def)

		# compute distance


		map_after = self.freeMap_size
		map_gain = map_after - map_before

		reward = self.compute_reward(D, map_gain)


		if(self.step_count >= self.max_actions or reward == 0):
			done = True
			self.step_count = 0
		else:
			done = False


		self.step_count += 1


		new_state = np.asarray([self.robot_pose, self.frontier, self.map])

		return new_state, reward, done


	# compute the reward for this action
	def compute_reward(self,D, map_gain):

		if(self.crash == 0):
			map_reward = 0.1*float(map_gain) #/float(self.freeMap_size)
			re = D + map_reward

			print("Map Reward = %f" % map_reward)
			print("Distance Reward = %f" % D)

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



def feedback_linearization(Ux, Uy, theta_n):
	d = 0.2

	vx = cos(theta_n) * Ux + sin(theta_n) * Uy
	w = -(sin(theta_n) * Ux)/ d + (cos(theta_n) * Uy) / d

	return vx, w
def pot_att(x,y,px,py):
	D = sqrt((px-x)**2 + (py-y)**2)
	K = 0.2
	D_safe = 10.0

	if(D > D_safe):
		Ux = - D_safe*K*(x - px)/D
		Uy = - D_safe*K*(y - py)/D
		U_a = [Ux, Uy]
	else:
		Ux = - K*(x - px)
		Uy = - K*(y - py)
		U_a = [Ux, Uy]

	return U_a

def pot_rep(theta_n, D, alfa):
	K = 1.0
	D_safe = 1.5

	if( D > D_safe):
		Ux = 0
		Uy = 0
		U_r = [Ux, Uy]
	else:

		grad_x = - cos(alfa*pi/180.0 + theta_n)
		grad_y = - sin(alfa*pi/180.0 + theta_n)
		Ux = K * (1.0/D_safe - 1.0/D) * (1.0/D**2) * grad_x 
		Uy = K * (1.0/D_safe - 1.0/D) * (1.0/D**2) * grad_y

		U_r = [Ux, Uy]

	return U_r

def min_dist(x_n,y_n,theta_n,laser):
	d_min = laser[0]
	alfa = 0
	for i in range(len(laser)):
		if(laser[i] < d_min):
			d_min = laser[i]
			alfa = i

	sx = (cos(theta_n)*(d_min*cos(np.deg2rad(alfa - 180))) + sin(theta_n)*(d_min*sin(np.deg2rad(alfa - 180)))) + x_n
	sy = (-sin(theta_n)*(d_min*cos(np.deg2rad(alfa - 180))) + cos(theta_n)*(d_min*sin(np.deg2rad(alfa - 180)))) + y_n	

	obs_pos = [sx, sy]
	return d_min, alfa, obs_pos



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
	traj_msg.filter_path_n_average = 5


	return traj_msg
