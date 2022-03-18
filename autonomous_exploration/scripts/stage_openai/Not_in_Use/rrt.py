#!/usr/bin/env python
########################################
'''               RRT               '''
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
import matplotlib.pyplot as plt
import numpy as np
import random
from math import atan2, cos, sin, sqrt, hypot, floor, ceil


 
########################################
'''             RRT Class            '''
########################################
class RRT:
    class Node:
        """
        Class Node
        """
        def __init__(self, x, y):
            self.x = x
            self.y = y
            self.path_x = []
            self.path_y = []
            self.parent = None

    def __init__(self, start, goal, obstacle_list, map_size, step_size=2.0, path_frac=0.1, max_iter=500):
        self.start = self.Node(start[0], start[1])
        self.end_point = self.Node(goal[0], goal[1])
        self.min_rand = map_size[0]
        self.max_rand = map_size[1]
        self.step_size = step_size
        self.path_frac = path_frac
        self.max_iter = max_iter
        self.obstacle_list = obstacle_list
        self.node_list = []

    def planning(self):

        self.node_list = [self.start]
        for i in range(self.max_iter):
            q_rnd = self.get_random()
            q_new = self.extend_rrt(q_rnd)

            if dist([self.node_list[-1].x, self.node_list[-1].y],[self.end_point.x,self.end_point.y]) <= self.step_size:
                q_final = self.step(self.node_list[-1], self.end_point)
                if self.check_collision(q_final):
                    return self.final_path(len(self.node_list) - 1)

        return None  # cannot find path


    def extend_rrt(self,q_rand):
        q_near = self.find_qnear(q_rand)
        q_new = self.step(q_near, q_rand)

        if self.check_collision(q_new):
            self.node_list.append(q_new)
            return q_new
        else:
            return None


    def step(self, q1, q2):
        step_len = self.step_size

        q_new = self.Node(q1.x, q1.y)
        d = dist([q2.x,q2.y],[q_new.x,q_new.y])
        theta = atan2((q2.y-q_new.y),(q2.x-q_new.x))

        q_new.path_x = [q_new.x]
        q_new.path_y = [q_new.y]

        if step_len > d:
            step_len = d

        n = int(floor(step_len / self.path_frac))

        for _ in range(n):
            q_new.x += self.path_frac * cos(theta)
            q_new.y += self.path_frac * sin(theta)
            q_new.path_x.append(q_new.x)
            q_new.path_y.append(q_new.y)

        d = dist([q2.x,q2.y],[q_new.x,q_new.y])
        if d <= self.path_frac:
            q_new.path_x.append(q2.x)
            q_new.path_y.append(q2.y)
            q_new.x = q2.x
            q_new.y = q2.y

        q_new.parent = q1

        return q_new

    def final_path(self, goal_ind):
        path = [[self.end_point.x, self.end_point.y]]
        node = self.node_list[goal_ind]
        while node.parent is not None:
            path.append([node.x, node.y])
            node = node.parent
        path.append([node.x, node.y])

        return path

    def get_random(self):
        rnd = self.Node(random.uniform(self.min_rand, self.max_rand), random.uniform(self.min_rand, self.max_rand))
        return rnd

    def find_qnear(self,q_rnd):
        dlist = [(q.x - q_rnd.x)**2 + (q.y - q_rnd.y)**2
                 for q in self.node_list]
        minind = dlist.index(min(dlist))

        return self.node_list[minind]


    def check_collision(self, q):
        obstacleList = self.obstacle_list

        if q is None:
            return False

        for (cx, cy, size) in obstacleList:
            dx_list = [cx - x for x in q.path_x]
            dy_list = [cy - y for y in q.path_y]
            d_list = [dx * dx + dy * dy for (dx, dy) in zip(dx_list, dy_list)]

            if min(d_list) <= 3*size**2:
                return False  # collision

        return True  # safe



    def draw_graph(self, rnd=None):
        plt.clf()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])
        if rnd is not None:
            plt.plot(rnd.x, rnd.y, "^k")

        traj = []
        for node in self.node_list:
            if node.parent:
                plt.plot(node.path_x, node.path_y, "-g")
                traj.append([node.path_x, node.path_y])

        # new_path2(traj,pub)


        for (ox, oy, size) in self.obstacle_list:
            plot_circle(ox, oy, size)

        plt.plot(self.start.x, self.start.y, "xr")
        plt.plot(self.end_point .x, self.end_point .y, "xr")
        plt.axis("equal")
        plt.axis([-50, 50, -50, 50])
        plt.grid(True)
        # plt.pause(0.01)

def plot_circle(x, y, size, color="-k"):  # pragma: no cover
    deg = list(range(0, 360, 5))
    deg.append(0)
    xl = [x + size * cos(np.deg2rad(d)) for d in deg]
    yl = [y + size * sin(np.deg2rad(d)) for d in deg]
    plt.plot(xl, yl, color)


########################################
'''           Publish Path           '''
########################################
def new_path2(traj, pub):
    if not pub:
        raise AssertionError("pub is not valid:%s".format(pub))

    path = Path()

    print(len(traj[0]))

    for j in range(len(traj[0][:][:])):
        for i in range(len(traj[0][0][:])):
            # for j in range(len(traj))
            pose = PoseStamped()
            pose.header.frame_id = "/odom"
            pose.header.stamp = rospy.Time.now()

            pose.pose.position.x = traj[j][0][i]
            pose.pose.position.y = traj[j][1][i]
            pose.pose.position.z = 0

            path.poses.append(pose)

    path.header.frame_id = "/odom"
    path.header.stamp = rospy.Time.now()
    pub.publish(path)



########################################
'''      Dist between two points     '''
########################################
def dist(p1,p2): 
    return sqrt((p1[0]-p2[0])*(p1[0]-p2[0])+(p1[1]-p2[1])*(p1[1]-p2[1]))



########################################
'''            Callbacks             '''
########################################
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
    global goal
    goal = ((data.point.x),(data.point.y))



def callback_map(msg):
    global mapa
    mapa = np.asarray(msg.data, dtype=np.int8).reshape(msg.info.height, msg.info.width)
    n = len(mapa)


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




########################################
'''           Discret Map            '''
########################################
def map(map_name, fact, resol):
    rospack = rospkg.RosPack()
    path = rospack.get_path('MotionPlanner_RRT_Astar')
    image_path = path + '/worlds/' + map_name
    image = img.imread(image_path)
    # image.setflags(write=1)

    idx = []

    M = np.zeros((len(image),len(image)))
    for i in range(len(image)):
        for j in range(len(image)):
            if(image[i,j,0] == 255 and image[i,j,1] == 255 and image[i,j,2] == 255):
                M[i,j] = 0
            else:
                M[i,j] = 1
                idx.append([float(j*resol)-fact, -float(i*resol)+fact,  0.3])
                idx.append([float(j*resol)-fact+0.15, -float(i*resol)+fact+0.15,  0.3])
                idx.append([float(j*resol)-fact-0.15, -float(i*resol)+fact-0.15,  0.3])

    return M, idx


########################################
'''           Main Function          '''
########################################
def run():
    global robot_states, goal, mapa, idx

    # states - x,y, theta
    robot_states = [0.0, 0.0, 0.0]
    # control msg
    vel_msg = Twist()
    # Control class
    controlador = control()

    ## ROS STUFFS
    rospy.init_node("rrt", anonymous=True)

    # Publishers
    pub_cmd_vel = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
    pub_path = rospy.Publisher("/final_path", Path, queue_size=10)
    pub_path2 = rospy.Publisher("/rrt_path", Path, queue_size=10)

    # Subscribers
    rospy.Subscriber('/base_pose_ground_truth', Odometry, callback_pose)
    rospy.Subscriber('/clicked_point', PointStamped, callback_goalPoint)
    rospy.Subscriber('/map', OccupancyGrid, callback_map)

    # routine frequency
    rate = rospy.Rate(5)

    ####### Get Map
    mapa = []
    M, obstacle_list = map('map_obstacle3.bmp', 50, 1.0)

    ####### RRT - class
    # Start point
    start = (robot_states[0],robot_states[1])
    goal = []
    max_samples = 5000

    flag_start = True
    
    while not rospy.is_shutdown():

        start = ((robot_states[0]),(robot_states[1]))
        if goal and obstacle_list and flag_start==True:
            flag_start = False

            rrt_path = RRT(start=start,goal=goal,map_size=[-50,50],obstacle_list=obstacle_list,max_iter=max_samples)

            print("Planning")
            path = rrt_path.planning()

            if path is None:
                print("Cannot find path")
            else:
                print("found path!!")

                new_traj = np.zeros((len(path),2))
                j = 0
                for i in range(len(path)-1,-1,-1):
                    new_traj[j,0] = path[i][0]
                    new_traj[j,1] = path[i][1]
                    j+=1

                # rrt_path.draw_graph(pub_path2)

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

                        print("Robot Pos = [%f, %f]\n Target Pos = [%f, %f]\n Distancy = %f\n\n" % (robot_states[0],robot_states[1],new_traj[i,0],new_traj[i,1],D))

                        vel_msg.linear.x, vel_msg.angular.z = controlador.control_([new_traj[i,0],new_traj[i,1]],robot_states)
                        pub_cmd_vel.publish(vel_msg)

            flag_start = True
            goal = []

        else:
            print("wainting goal")
    	rate.sleep()



########################################
'''            Main Routine          '''
########################################
if __name__ == '__main__':
    try:
        run()
    except rospy.ROSInterruptException:
        pass
