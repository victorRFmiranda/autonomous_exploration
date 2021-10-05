#!/usr/bin/env python
########################################
'''               A*               '''
########################################
'''
@author: Victor R. F. Miranda
@institute: Universidade Federal de Minas Gerais (UFMG)
@contact: victormrfm@ufmg.br
@course: PhD in Electrical Engineering
'''
import rospy
import rospkg
from tf.transformations import euler_from_quaternion


# ros-msgs
from nav_msgs.msg import Odometry, OccupancyGrid, Path
from geometry_msgs.msg import Twist, PointStamped, PoseStamped

# python
import matplotlib.image as img
import numpy as np
import math
import time





## class for the Nodes in the Grid
class Node():
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position


## Compute a list of nodes of a path from the given start to the given target
def Astar(maze, start, target):
    

    # Create start and end node
    start_node = Node(None, start)
    start_node.g = start_node.h = start_node.f = 0
    end_node = Node(None, target)
    end_node.g = end_node.h = end_node.f = 0

    # Initialize both open and closed list
    open_list = []
    closed_list = []

    # Add the start node
    open_list.append(start_node)

    # Loop until find the end
    while len(open_list) > 0:


        # Get the current node
        current_node = open_list[0]
        current_index = 0
        for index, item in enumerate(open_list):
            if item.f < current_node.f:
                current_node = item
                current_index = index

        # Remove current from open list, add to closed list
        open_list.pop(current_index)
        closed_list.append(current_node)

        # Found the goal
        if current_node == end_node:
            path = []
            current = current_node
            while current is not None:
                path.append(current.position)
                current = current.parent
            return path[::-1] # Return reversed path

        # Generate children
        children = []
        for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]: # Adjacent squares

            # Get node position
            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

            # Make sure within range
            if node_position[0] > (len(maze) - 1) or node_position[0] < 0 or node_position[1] > (len(maze[len(maze)-1]) -1) or node_position[1] < 0:
                continue

            # Make sure walkable terrain
            if maze[node_position[0]][node_position[1]] != 0:
                continue

            # Create new node
            new_node = Node(current_node, node_position)

            # Append
            children.append(new_node)

        # Loop through children
        for child in children:


            # Child is on the closed list
            for closed_child in closed_list:
                if child == closed_child:
                    continue

            # Create the f, g, and h values
            child.g = current_node.g + 1
            child.h = ((child.position[0] - end_node.position[0]) ** 2) + ((child.position[1] - end_node.position[1]) ** 2)
            child.f = child.g + child.h

            # Child is already in the open list
            for open_node in open_list:
                if child == open_node and child.g > open_node.g:
                    continue

            # Add the child to the open list
            open_list.append(child)

    return open_list




##    Rotina callback para a obtencao da pose do robo
def callback_pose(data):
    global robot_states

    robot_states[0] = data.pose.pose.position.x  # posicao 'x' do robo no mundo 
    robot_states[1] = data.pose.pose.position.y  # posicao 'y' do robo no mundo 

    x_q = data.pose.pose.orientation.x
    y_q = data.pose.pose.orientation.y
    z_q = data.pose.pose.orientation.z
    w_q = data.pose.pose.orientation.w
    euler = euler_from_quaternion([x_q, y_q, z_q, w_q])

    robot_states[2] = euler[2]  # orientacao do robo no mundo 
            
    return


def callback_goalPoint(data):
    global goal_cb
    goal_cb = [data.point.x,data.point.y]
    # goal = (int(round((data.point.x+50)/0.025)),int(round((data.point.y+50)/0.025)))


def callback_map(msg):
    global mapa, resol, size, width, height, origem_map
    resol = msg.info.resolution
    width = msg.info.width
    height = msg.info.height
    origem_map = [msg.info.origin.position.x,msg.info.origin.position.y]
    size = [origem_map[0]+(msg.info.width * resol),origem_map[1]+(msg.info.height * resol)]


    mapa = np.asarray(msg.data, dtype=np.int8).reshape(msg.info.height, msg.info.width)
    mapa = np.where(mapa==0,0,1)
    # mapa = data.data


########################################
'''      Dist between two points     '''
########################################
def dist(p1,p2): 
    return math.sqrt((p1[0]-p2[0])*(p1[0]-p2[0])+(p1[1]-p2[1])*(p1[1]-p2[1]))


########################################
'''           Control Class          '''
########################################
class control:
    def __init__(self):
        self.d = 0.1
        self.k = 1

    def control_(self,pos_curve, robot_states):

        Ux = self.k * (pos_curve[0] - robot_states[0])
        Uy = self.k * (pos_curve[1] - robot_states[1])

        return self.feedback_linearization(Ux,Uy,robot_states[2])

    def feedback_linearization(self,Ux, Uy, theta_n):

        vx = math.cos(theta_n) * Ux + math.sin(theta_n) * Uy
        w = -(math.sin(theta_n) * Ux)/ self.d  + (math.cos(theta_n) * Uy) / self.d 

        return vx, w



########################################
'''           Publish Path           '''
########################################
def new_path(traj, pub):
    if not pub:
        raise AssertionError("pub is not valid:%s".format(pub))

    path = Path()

    for i in range(len(traj[0])):
        pose = PoseStamped()
        pose.header.frame_id = "/map"
        pose.header.stamp = rospy.Time.now()

        pose.pose.position.x = traj[0][i]
        pose.pose.position.y = traj[1][i]
        pose.pose.position.z = 0

        path.poses.append(pose)

    path.header.frame_id = "/map"
    path.header.stamp = rospy.Time.now()
    pub.publish(path)




########################################
'''           Discret Map            '''
########################################
def map(map_name):
    rospack = rospkg.RosPack()
    path = rospack.get_path('MotionPlanner_RRT_Astar')
    image_path = path + '/worlds/' + map_name
    image = img.imread(image_path)
    #image.setflags(write=1)

    M = np.zeros((len(image),len(image)))
    for i in range(len(image)):
        for j in range(len(image)):
            if(image[i,j,0] == 255 and image[i,j,1] == 255 and image[i,j,2] == 255):
                M[i,j] = 0
            else:
                M[i,j] = 1

    return M


########################################
'''           Main Function          '''
########################################
def run():
    global robot_states, goal, goal_cb, mapa, resol, size, width, height, origem_map

    # states - x,y, theta
    robot_states = [0.0, 0.0, 0.0]
    # control msg
    vel_msg = Twist()
    # Control class
    controlador = control()

    ## ROS STUFFS
    rospy.init_node("AStar", anonymous=True)

    # Publishers
    pub_cmd_vel = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
    pub_path = rospy.Publisher("/final_path", Path, queue_size=10)

    # Subscribers
    rospy.Subscriber('/base_pose_ground_truth', Odometry, callback_pose)
    rospy.Subscriber('/clicked_point', PointStamped, callback_goalPoint)
    rospy.Subscriber('/map', OccupancyGrid, callback_map)

    # routine frequency
    rate = rospy.Rate(5)

    ####### Get Map
    resol = 0
    size = [0,0]
    width = 0
    height = 0
    origem_map = [0,0]
    goal_cb = []
    mapa = []
    M = map('map_obstacle2.bmp')

    ####### RRT - class
    # Start point
    # start = [0.0, 0.0]
    # start = (robot_states[0],robot_states[1])
    goal = []
    max_samples = 100
    # planner = rrt(start, goal, max_samples)
    s = []

    time.sleep(1)

    while not rospy.is_shutdown():
        # start = (int(round((robot_states[0]+50)/0.025)),int(round((robot_states[1]+50)/0.025)))
        start = (int(round((robot_states[0]-origem_map[0]-resol/2.0)/resol)),int(round((robot_states[1]-origem_map[1]-resol/2.0)/resol)))
        # start = (int(round((-robot_states[1]*resol)+size[1])),int(round((robot_states[0]*resol)+size[0])))
        # print("resol",resol)
        # print("width",width)
        # print("height",height)
        # print("size",size)
        # print("origem",origem_map)
        print("Start=",start)
        if goal_cb:
            goal = (int(round((goal_cb[0]-origem_map[0]-resol/2.0)/resol)),int(round((goal_cb[1]-origem_map[1]-resol/2.0)/resol)))
            print("goal=",goal)
            print("computing map")
            path = Astar(mapa, start, goal)

            vec_path = np.zeros((len(path),2))
            for i in range(len(path)):
                s = list(path[i])
                vec_path[i,:] = list(path[i])
                # vec_path[i,0] = vec_path[i,0]*0.025 - 50
                # vec_path[i,1] = vec_path[i,1]*0.025 - 50
                vec_path[i,0] = origem_map[0] + (vec_path[i,0]*resol + resol/2.0)
                vec_path[i,1] = origem_map[1] + (vec_path[i,1]*resol + resol/2.0)

            t_x = []
            t_y = []
            t_y = vec_path[:,1]
            t_x = vec_path[:,0]

            new_path([t_x,t_y],pub_path)

            print(path)

            goal = []

            # Controle
            for i in range(len(t_x)):
                t_init = rospy.get_time()
                D = 1000
                while(D > 0.2 and not rospy.is_shutdown()):
                    # D = math.sqrt((t_y[i]-robot_states[1])**2+(t_x[i]-robot_states[0])**2)
                    D = dist([t_x[i],t_y[i]],[robot_states[0],robot_states[1]])
                    t = rospy.get_time() - t_init

                    print("Robot Pos = [%f, %f]\n Target Pos = [%f, %f]\n Distancy = %f\n\n" % (robot_states[0],robot_states[1],t_x[i],t_y[i],D))

                    vel_msg.linear.x, vel_msg.angular.z = controlador.control_([t_x[i],t_y[i]],robot_states)
                    pub_cmd_vel.publish(vel_msg)

        # else:
        #     print("wainting goal")
    	# print("Robot pose: x = %f, y = %f, yaw = %f\n" % (robot_states[0],robot_states[1],robot_states[2]))
    	rate.sleep()



########################################
'''            Main Routine          '''
########################################
if __name__ == '__main__':
    try:
        run()
    except rospy.ROSInterruptException:
        pass
