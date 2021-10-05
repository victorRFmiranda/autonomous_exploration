#!/usr/bin/env python
########################################
'''               RRT*               '''
########################################
'''
@author: Victor R. F. Miranda
@institute: Universidade Federal de Minas Gerais (UFMG)
@contact: victormrfm@ufmg.br
@course: PhD in Electrical Engineering
'''


# Ros-libs
import rospy
import rospkg
from tf.transformations import euler_from_quaternion

# Ros-msgs
from nav_msgs.msg import Odometry, OccupancyGrid, Path
from geometry_msgs.msg import Twist, PointStamped, PoseStamped


# python
import os
import time
import matplotlib.image as img
import matplotlib.pyplot as plt
import numpy as np
import random
from math import pi, atan2, tan, cos, sin, sqrt, hypot, floor, ceil, log





########################################
'''             RRT* Class           '''
########################################
class RRTStar:
	class Node:
		"""
		Class Node
		"""
		def __init__(self, x, y, theta):
			self.x = x
			self.y = y
			self.theta = theta
			self.path_x = []
			self.path_y = []
			self.parent = None
			self.cost = 0.0

	def __init__(self, start, goal, resol_map, occupancy_map, obstacle_list, map_size, step_size=0.5, dt=0.2, max_iter=500, connect_circle_dist=10.0, search_until_max_iter=False):
		self.start = self.Node(start[0], start[1], start[2])
		self.end_point = self.Node(goal[0], goal[1], 0.0)
		self.min_x = map_size[0][0]
		self.max_x = map_size[1][0]
		self.min_y = map_size[0][1]
		self.max_y = map_size[1][1]

		self.step_size = step_size
		self.dt = dt
		self.max_iter = max_iter
		self.obstacle_list = obstacle_list
		self.occ_map = occupancy_map
		self.resol_map = resol_map
		self.node_list = []

		self.connect_circle_dist = connect_circle_dist
		self.search_until_max_iter = search_until_max_iter

		# robot velocity for nonholonomic paths
		self.uV = 0.5
		self.uW = [pi/-6.0, pi/-12.0, 0.0, pi/6.0, pi/12.0]

	def planning(self):
		self.node_list = [self.start]
		for i in range(self.max_iter):
			print("Iter:", i, ", number of nodes:", len(self.node_list))
			q_rnd = self.get_random()
			q_new = self.extend_rrt(q_rnd)

			if ((not self.search_until_max_iter) and q_new):  # if reaches goal
				last_index = self.search_best_goal_node()
				if last_index is not None:
					aaa = self.final_path(len(self.node_list) - 1)
					return aaa

		print("reached max iteration")

		last_index = self.search_best_goal_node()
		if last_index is not None:
			aaa = self.final_path(len(self.node_list) - 1)
			return aaa

		return None  # cannot find path


	def extend_rrt(self,q_rand):
		q_near = self.find_qnear(q_rand)
		q_new = self.step(q_near, q_rand)

		q_new.cost = q_near.cost + dist([q_new.x,q_new.y],[q_near.x,q_near.y])

		if self.check_collision(q_new):
			self.node_list.append(q_new)

			near_inds = self.find_near_nodes(q_new)
			node_with_updated_parent = self.choose_parent(q_new, near_inds)
			if node_with_updated_parent:
				# self.rewire(node_with_updated_parent, near_inds)
				self.node_list.append(node_with_updated_parent)
			else:
				self.node_list.append(q_new)


			return q_new
		else:
			return None


	# Computes q_new from q_near to q_rand with a distance step
	def step(self,q1,q2):

		xr=[]
		yr=[]
		thetar=[]
		# 
		for j in self.uW:
			(x,y,theta)=self.trajectory(q1.x,q1.y,q1.theta,j)
			xr.append(x)
			yr.append(y)
			thetar.append(theta)

		# find the best traj from q1 to q2
		dmin = dist([q2.x,q2.y],[xr[0][-1],yr[0][-1]])
		near = 0
		for i in range(1,len(xr)):
			d = dist([q2.x,q2.y],[xr[i][-1],yr[i][-1]])
			if d < dmin:
				dmin= d
				near = i

		# Define q_new
		q_new = self.Node(xr[near][-1],yr[near][-1],thetar[near][-1])
		q_new.parent = q1
		q_new.path_x = xr[near]
		q_new.path_y = yr[near]

		return q_new


	# generate trajectory from equations of motion         
	def trajectory(self,xi,yi,thetai,ori_vec):
		(x,y,theta)=([],[],[])
		x.append(xi)
		y.append(yi)
		theta.append(thetai)
		p = self.step_size/self.dt
		for i in range(1,int(p)):
			theta.append(theta[i-1]+(self.uV*tan(ori_vec))*self.dt)
			x.append(x[i-1]+self.uV*cos(theta[i-1])*self.dt)
			y.append(y[i-1]+self.uV*sin(theta[i-1])*self.dt)    

		return (x,y,theta)


	# get final path
	def final_path(self, goal_ind):
		node = self.node_list[goal_ind]
		path = []	

		# Compute a smooth path to the final node
		xr=[]
		yr=[]
		thetar=[]
		for j in self.uW:
			(x,y,theta)=self.trajectory(node.x,node.y,node.theta,j)
			xr.append(x)
			yr.append(y)
			thetar.append(theta)
		dmin = dist([self.end_point.x,self.end_point.y],[xr[0][-1],yr[0][-1]])
		near = 0
		for i in range(1,len(xr)):
			d = dist([self.end_point.x,self.end_point.y],[xr[i][-1],yr[i][-1]])
			if d < dmin:
				dmin= d
				near = i
		self.end_point.path_x = xr[near]
		self.end_point.path_y = yr[near]

		N = len(self.end_point.path_x) - 1
		while N >=0:
			path.append([self.end_point.path_x[N],self.end_point.path_y[N]])
			N -=1


		# Compute the paths to the others nodes.
		while node.parent is not None:
			n = len(node.path_x) - 1
			while n >=0:
				path.append([node.path_x[n],node.path_y[n]])
				n -= 1
			node = node.parent
		path.append([node.x, node.y])

		return path


	# Return all nodes near to the new_node, considering a ball
	def find_near_nodes(self, new_node):
		nnode = len(self.node_list) + 1
		r = self.step_size
		dist_list = [(node.x - new_node.x)**2 + (node.y - new_node.y)**2
		             for node in self.node_list]
		near_inds = [dist_list.index(i) for i in dist_list if i <= r**2]

		return near_inds


	# Computes the cheapest point to new_node contained in the list	near_inds and set such a node as the parent of new_node.
	def choose_parent(self, new_node, near_inds):
		if not near_inds:
			return None

		# search nearest cost in near_inds
		costs = []
		for i in near_inds:
			near_node = self.node_list[i]
			t_node = self.step(near_node, new_node)
			if t_node and self.check_collision(t_node):
				costs.append(self.calc_new_cost(near_node, new_node))
			else:
				costs.append(float("inf"))  # the cost of collision node
		min_cost = min(costs)

		if min_cost == float("inf"):
			print("There is no good path.(min_cost is inf)")
			return None

		min_ind = near_inds[costs.index(min_cost)]
		new_node = self.step(self.node_list[min_ind], new_node)
		new_node.cost = min_cost

		return new_node


	# Re-assing the parents of the nodes in case of a cheaper cost
	def rewire(self, new_node, near_inds):
		for i in near_inds:
			near_node = self.node_list[i]
			edge_node = self.step(new_node, near_node)
			if not edge_node:
				continue
			edge_node.cost = self.calc_new_cost(new_node, near_node)

			no_collision = self.check_collision(edge_node)
			improved_cost = near_node.cost > edge_node.cost

			if no_collision and improved_cost:
				near_node.x = edge_node.x
				near_node.y = edge_node.y
				near_node.cost = edge_node.cost
				near_node.path_x = edge_node.path_x
				near_node.path_y = edge_node.path_y
				near_node.parent = edge_node.parent
				self.propagate_cost_to_leaves(new_node)

	def search_best_goal_node(self):
		dist_to_goal_list = [dist([n.x, n.y],[self.end_point.x,self.end_point.y]) for n in self.node_list]
		goal_inds = [dist_to_goal_list.index(i) for i in dist_to_goal_list	if i <= self.step_size]

		safe_goal_inds = []
		for goal_ind in goal_inds:
			t_node = self.step(self.node_list[goal_ind], self.end_point)
			if self.check_collision(t_node):
				safe_goal_inds.append(goal_ind)

		if not safe_goal_inds:
			return None

		min_cost = min([self.node_list[i].cost for i in safe_goal_inds])
		for i in safe_goal_inds:
			if self.node_list[i].cost == min_cost:
				return i

		return None



	def calc_new_cost(self, from_node, to_node):
		d = dist([to_node.x,to_node.y],[from_node.x,from_node.y])
		return from_node.cost + d

	def propagate_cost_to_leaves(self, parent_node):

		for node in self.node_list:
			if node.parent == parent_node:
				node.cost = self.calc_new_cost(parent_node, node)
				self.propagate_cost_to_leaves(node)


	def get_random(self):
		rnd = self.Node(random.uniform(self.min_x, self.max_x), random.uniform(self.min_y, self.max_y), random.uniform (0, pi))
		r = dist([self.end_point.x,self.end_point.y],[self.start.x,self.start.y])
		# rnd = self.Node(random.uniform(self.start.x, self.end_point.x), random.uniform(self.start.y, self.end_point.y), random.uniform (0, pi))
		d = dist([self.end_point.x,self.end_point.y],[rnd.x,rnd.y])
		while (self.check_collision(rnd) == False and d < r/3.0):
		# while (self.check_collision(rnd) == False):
			rnd = self.Node(random.uniform(self.min_x, self.max_x), random.uniform(self.min_y, self.max_y), random.uniform (0, pi))
			# rnd = self.Node(random.uniform(self.start.x, self.end_point.x), random.uniform(self.start.y, self.end_point.y), random.uniform (0, pi))
			d = dist([self.end_point.x,self.end_point.y],[rnd.x,rnd.y])
		return rnd

	def find_qnear(self,q_rnd):
		dlist = [(q.x - q_rnd.x)**2 + (q.y - q_rnd.y)**2 for q in self.node_list]
		minind = dlist.index(min(dlist))

		return self.node_list[minind]


	def check_collision(self, q):
		occ_map = self.occ_map.data
		width = self.occ_map.info.width

		if q is None:
			return False

		w_list = []
		h_list = []

		if (q.path_x):
			w_list = np.array([(x - self.min_x - self.resol_map/2.0)/self.resol_map for x in q.path_x])
			h_list = np.array([(y - self.min_y - self.resol_map/2.0)/self.resol_map for y in q.path_y])
			w_list = np.around(w_list).astype(int)
			h_list = np.around(h_list).astype(int)
			for (w,h) in zip(w_list,h_list):
				# Check Safe distance
				for i in range(-4,5,1):
					for j in range(-4,5,1):
						if (occ_map[(h+i)*width + (w+j)] > 0):
							return False
		else:
			w = int(round( (q.x - self.min_x - self.resol_map/2.0)/self.resol_map ))
			h = int(round( (q.y - self.min_y - self.resol_map/2.0)/self.resol_map ))
			if(occ_map[h*width + w] != 0):
				return False 


		return True  # safe
		

	# Draw RRT
	def draw_graph(self):
		plt.clf()
		traj = []
		for node in self.node_list:
			if node.parent:
				plt.plot(node.path_x, node.path_y, "-g")
				traj.append([node.path_x, node.path_y])

		# HIGH LOADING - can be removed if needed 
		for (ox, oy, size) in self.obstacle_list:
			plot_circle(ox, oy, size)

		plt.plot(self.start.x, self.start.y, "xr")
		plt.plot(self.end_point .x, self.end_point .y, "xr")
		plt.axis("equal")
		plt.axis([-50, 50, -50, 50])
		plt.grid(True)




########################################
'''      Dist between two points     '''
########################################
def dist(p1,p2): 
	return ((p1[0]-p2[0])**2 +(p1[1]-p2[1])**2)**(0.5)


########################################
'''         Plot Obstacles           '''
########################################
def plot_circle(x, y, size, color="-k"):
	deg = list(range(0, 360, 5))
	deg.append(0)
	xl = [x + size * cos(np.deg2rad(d)) for d in deg]
	yl = [y + size * sin(np.deg2rad(d)) for d in deg]
	plt.plot(xl, yl, color)




########################################
'''            Callbacks             '''
########################################
def callback_pose(data):
    global robot_states

    robot_states[0] = data.pose.pose.position.x  # robot pos x
    robot_states[1] = data.pose.pose.position.y  # robot pos y

    x_q = data.pose.pose.orientation.x
    y_q = data.pose.pose.orientation.y
    z_q = data.pose.pose.orientation.z
    w_q = data.pose.pose.orientation.w
    euler = euler_from_quaternion([x_q, y_q, z_q, w_q])

    robot_states[2] = euler[2]  # robot orientation
            
    return


def callback_goalPoint(data):
    global goal
    goal = ((data.point.x),(data.point.y))



def callback_obst(msg):
	global size, origem_map, width, height, resol, occ_map
	occ_map = msg
	resol = msg.info.resolution
	width = msg.info.width
	height = msg.info.height
	origem_map = [msg.info.origin.position.x,msg.info.origin.position.y]
	size = [origem_map[0]+(msg.info.width * resol),origem_map[1]+(msg.info.height * resol)]



########################################
'''           Control Class          '''
########################################
class control:
    def __init__(self):
        self.d = 0.2
        self.k = 5

    def control_(self,pos_curve, robot_states):

        Ux = self.k * (pos_curve[0] - robot_states[0])
        Uy = self.k * (pos_curve[1] - robot_states[1])

        return self.feedback_linearization(Ux,Uy,robot_states[2])

    def feedback_linearization(self,Ux, Uy, theta_n):

        vx = cos(theta_n) * Ux + sin(theta_n) * Uy
        w = -(sin(theta_n) * Ux)/ self.d  + (cos(theta_n) * Uy) / self.d 

        return vx, w



########################################
'''           Publish Path           '''
########################################
def new_path(traj, pub):
    if not pub:
        raise AssertionError("pub is not valid:%s".format(pub))

    path = Path()

    for i in range(len(traj)):
        pose = PoseStamped()
        pose.header.frame_id = "/odom"
        pose.header.stamp = rospy.Time.now()

        pose.pose.position.x = traj[i,0]
        pose.pose.position.y = traj[i,1]
        pose.pose.position.z = 0

        path.poses.append(pose)

    path.header.frame_id = "/odom"
    path.header.stamp = rospy.Time.now()
    pub.publish(path)



def compute_obstacles(width,height,resol,origem_map,msg):
	global robot_states

	obst = []
	front_vect = []
	r = (resol)
	# i for x, j for y
	for i in range(width):
		for j in range(height):
			if(msg.data[j*width + i] > 0):
				xs = origem_map[0] + (i * resol + resol/2.0)
				ys = origem_map[1] + (j * resol + resol/2.0)
				obst.append([xs,ys, r])



	return obst



########################################
'''           Main Routine          '''
########################################
def run():
	global robot_states, goal, size, origem_map, width, height, resol, occ_map

	# states - x,y, theta
	robot_states = [0.0, 0.0, 0.0]
	goal = np.array([])
	size = [0.0,0.0]
	origem_map = [0.0,0.0]
	max_samples = 10000
	occ_map = OccupancyGrid()
	width = 0
	height = 0
	resol = 0
	# control msg
	vel_msg = Twist()
	# Control class
	controlador = control()

	## ROS STUFFS
	rospy.init_node("rrt", anonymous=True)

	# Publishers
	pub_cmd_vel = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
	pub_path = rospy.Publisher("/final_path", Path, queue_size=10)

	# Subscribers
	rospy.Subscriber('/base_pose_ground_truth', Odometry, callback_pose)
	# rospy.Subscriber('/odom', Odometry, callback_pose)
	rospy.Subscriber('/map', OccupancyGrid, callback_obst)
	# rospy.Subscriber('/scan', LaserScan, callback_lidar)

	
	rospy.Subscriber('/clicked_point', PointStamped, callback_goalPoint)

	# routine frequency
	rate = rospy.Rate(5)


	####### RRT - class
	goal = []
	flag_start = True

	time.sleep(2)


	while not rospy.is_shutdown():
		# define start point
		start = ((robot_states[0]),(robot_states[1]), (robot_states[2]))


		if goal and flag_start==True:
			obstacle_list = compute_obstacles(width,height,resol,origem_map,occ_map)

			flag_start = False

			# rrt_path = RRTStar(start=start,goal=goal,map_size=[origem_map,size],obstacle_list=obstacle_list,max_iter=max_samples,step_size = 1.0, dt=0.2)
			rrt_path = RRTStar(start=start,goal=goal,map_size=[origem_map,size],resol_map = resol, occupancy_map = occ_map, obstacle_list=obstacle_list,max_iter=max_samples,step_size = 4.0, dt=0.8)

			print("Planning")
			path = rrt_path.planning()


			if path is None:
				print("Cannot find path")
				rrt_path.draw_graph()
				plt.show()
			else:
				print("found path!!")

				new_traj = np.zeros((len(path),2))
				j = 0
				for i in range(len(path)-1,-1,-1):
					new_traj[j,0] = path[i][0]
					new_traj[j,1] = path[i][1]
					j+=1

				rrt_path.draw_graph()
				plt.plot([x for (x, y) in path], [y for (x, y) in path], '-r')
				plt.grid(True)
				plt.xlabel('X (m)')
				plt.ylabel('Y (m)')
				plt.show()

				new_path(new_traj,pub_path)

				# Control
				for i in range(len(new_traj)):
					t_init = rospy.get_time()
					D = 1000
					while(D > 0.1 and not rospy.is_shutdown()):
						D = dist([new_traj[i,0],new_traj[i,1]],[robot_states[0],robot_states[1]])
						t = rospy.get_time() - t_init

						# print("Robot Pos = [%f, %f]\n Target Pos = [%f, %f]\n Distancy = %f\n\n" % (robot_states[0],robot_states[1],new_traj[i,0],new_traj[i,1],D))

						vel_msg.linear.x, vel_msg.angular.z = controlador.control_([new_traj[i,0],new_traj[i,1]],robot_states)
						pub_cmd_vel.publish(vel_msg)


			flag_start = True
			goal = []

		else:
			print("wainting goal")
		rate.sleep()


########################################
'''            Start Code            '''
########################################
if __name__ == '__main__':
    run()