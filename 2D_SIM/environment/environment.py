#!/usr/bin/env python

import math
import numpy as np
import cv2

from pysim2d import pysim2d
from .environment_fitness import FitnessData
from .environment_node import Node
# from .lidar_to_grid_map import Map, generate_ray_casting_grid_map
from .localmap import Map
# from .create_map import localmap
from .getfrontier import getfrontier
from .dijkstra import Dijkstra
from .astar_start_end import searching_control
from sklearn.cluster import DBSCAN, KMeans


########################################
'''           Control Class          '''
########################################
class Control:
    def __init__(self):
        self.d = 0.1
        self.k = 100.0
        self.vr = 0.5

    def dist(self, p1,p2): 
        return ((p1[0]-p2[0])**2 +(p1[1]-p2[1])**2)**(0.5)

    def control_(self,pos_curve, robot_states):

        D = self.dist( robot_states, pos_curve )

        D_vec = [robot_states[0]-pos_curve[0],robot_states[1]-pos_curve[1]]
        grad_D = [D_vec[0]/(D + 0.000001), D_vec[1]/(D + 0.000001)]



        P = 0.5*(D**2)
        # G = -(2/pi)*atan(self.k*sqrt(P))
        G = -(2/math.pi)*math.atan(self.k*P)
        H = math.sqrt(1-G**2)

        Ux = self.vr * (G*grad_D[0] + H*grad_D[0])
        Uy = self.vr * (G*grad_D[1] + H*grad_D[1])

        return self.feedback_linearization(Ux,Uy,robot_states[2])


    def feedback_linearization(self,Ux, Uy, theta_n):

        vx = math.cos(theta_n) * Ux + math.sin(theta_n) * Uy
        w = -(math.sin(theta_n) * Ux)/ self.d  + (math.cos(theta_n) * Uy) / self.d 

        return vx, w

 
class Environment:
    """
    Class as a wrapper for the Simulation2D.
    """

    def __init__(self, path_to_world):
        """
        Constructor to initialize the environment.
        :param path_to_world: The path to the world which should be selected.
        """
        self._env = pysim2d.pysim2d()
        self._fitness_data = FitnessData()
        self._cluster_size = 1
        self._observation_rotation_size = 64
        self._observation_rotation_use = False
        self._ditance_angle_to_end_use = False

        # self.m_height, self.m_width, self.m_resolution=12,12,0.1
        

        # self.m_height, self.m_width, self.m_resolution=12,12,0.1
        self.m_height, self.m_width, self.m_resolution=100,100,1.0
        # self.m_morigin=[self.m_width/2.0,self.m_height/2.0]
        self.control = Control()
        

        

        if not self._fitness_data.init(path_to_world + ".node"):
            print("Error: Load node file! -> " + path_to_world + ".node")
            exit(1)
        if not self._env.init(path_to_world + ".world"):
            print("Error: Load world file -> " + path_to_world + ".world")
            exit(1)

    def set_observation_rotation_size(self, size):
        """
        Set the vector size for the rotation.
        :param size: size of the observation rotation vector.
        :return:
        """
        if size < 8:
            print("Warn: Observation rotation size is to low -> set to 8!")

            self._observation_rotation_size = 8
        else:
            self._observation_rotation_size = size

    def use_ditance_angle_to_end(self, use=True):
        """
            Flag for using ditance from the robot to the target point and the angle between the robot
            orientation and the target.
            :param - use: True for using
            :return: 
        """
        self._ditance_angle_to_end_use = use

    def use_observation_rotation_size(self, use=True):
        """
        Flag for using the rotation observation vector. The vector is added to the laserscan observation. The rotation
        is decode in a vector like a compass. Dependent on the orientation of the robot to the target, the vector is
        filled with zero and a one by the orientation.

        Example:
        vector size:      8
        Target direction: 4
        vector values:    [0,0,0,0,1,0,0,0]

        1    2    3
             |
        0----+--->4
             |
        7    6    5

        :param use: True for using the rotation observation.
        :return:
        """
        self._observation_rotation_use = use


    ########################################
    '''         Compute Frontier         '''
    ########################################
    def _get_frontier(self):
        mapData, _ = self._get_map()
        # n_img = cv2.cvtColor(mapData,cv2.COLOR_GRAY2RGB)

        
        width, height = mapData.shape
        front_vect = []
        s = list([])
        s1 = list([])
        for i in range(width):
            for j in range(height):
                if(mapData[i,j] == 255):
                    if(i>2 and i < width-3 and j>2 and j < height-3):
                        s = np.array([ mapData[i-1,j-1], mapData[i-1,j], mapData[i-1,j+1],
                                    mapData[i,j-1], mapData[i,j+1],
                                    mapData[i+1,j-1], mapData[i+1,j], mapData[i+1,j+1] ])

                        s1 = np.array([ mapData[i-2,j-2], mapData[i-2,j-1], mapData[i-2,j], mapData[i-2,j+1], mapData[i-2,j+2],
                                    mapData[i-1,j-2], mapData[i-1,j-1], mapData[i-1,j], mapData[i-1,j+1], mapData[i-1,j+2],
                                    mapData[i,j-2], mapData[i,j-1], mapData[i,j], mapData[i,j+1], mapData[i,j+2],
                                    mapData[i+1,j-2], mapData[i+1,j-1], mapData[i+1,j], mapData[i+1,j+1], mapData[i+1,j+2],
                                    mapData[i+2,j-2], mapData[i+2,j-1], mapData[i+2,j], mapData[i+2,j+1], mapData[i+2,j+2] ])

                        if( (len(np.where(s==205)[0]) >= 3) and (len(np.where(s1==0)[0])<=0) ):
                            x = j
                            y = 100-i 

                            # n_img[i,j] = [255,0,0]

                            front_vect.append([x,y])

        # print("Frontier vec := ", front_vect)


        num_clusters = 4
        if(len(front_vect) > 4):
            # print(num_clusters)
            kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(front_vect)

        else: 
            size = len(front_vect)
            for i in range(num_clusters-size):
                front_vect.append(front_vect[0])

            kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(front_vect)



        frontiers = list([])
        angles = list([])
        pose = [self._env.get_robot_pose_x(),self._env.get_robot_pose_y(),self._env.get_robot_pose_orientation()]
        for i in range(num_clusters):
            sx = kmeans.cluster_centers_[i][0] - pose[0]
            sy = kmeans.cluster_centers_[i][1] - pose[1]
            ang_0 = math.atan2(sy,sx)
            diff = (pose[2] - ang_0) % (math.pi * 2)
            if diff >= math.pi:
                diff -= math.pi * 2

            angles.append(diff)

        frontiers = kmeans.cluster_centers_[np.argsort(angles)] 


        distances = []
        for k in front_vect:
            distances.append(self._distance(k[0],k[1],self._env.get_robot_pose_x(),self._env.get_robot_pose_y()))


        # f_l = np.asarray(frontiers).astype(int)
        # for k in (f_l):
        #     n_img[k[1],k[0]] = [0,0,255]

        # cv2.imshow('Mapa',n_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # return frontiers
        return frontiers, front_vect[np.argmin(distances)]



    ########################################
    '''         Compute Map         '''
    ########################################
    def _get_map(self):

        map_increase = 0.0

        lidar = self._get_observation_notNormalized()
        pose = [self._env.get_robot_pose_x(),self._env.get_robot_pose_y(),self._env.get_robot_pose_orientation()]

        pmap,map_increase  = self.Map.update_map(lidar,pose)

        # map_increase = 0
        # pmap, map_increase = self.Map_Teste.updatemap(lidar, pose)
        # print("pmap =: ", pmap.shape)

        return pmap, map_increase


    def _get_observation_notNormalized(self):
        """
        Get the observation from the laserscan and plus the observation rotation when activated.
        :return: Observation vector
        """
        size = self._env.observation_size()
        observation = []

        for i in range(size):
            observation.append(self._env.observation_lidar(i))

        return observation

    def _get_observation(self):
        """
        Get the observation from the laserscan and plus the observation rotation when activated.
        :return: Observation vector
        """
        size = self._env.observation_size()
        observation = []

        for i in range(size):
            observation.append(self._env.observation_at(i))

        return observation

    def _get_observation_min_clustered(self):
        """
        Get the observation size of the vector for clustering.
        :return: Observation vector size.
        """
        size = self._env.observation_min_clustered_size(self._cluster_size)
        observation = []

        for i in range(size):
            observation.append(self._env.observation_min_clustered_at(i, self._cluster_size))

        return observation

    def set_cluster_size(self, size):
        """
        Set the clustering size for the laserscan vector. The cluster size is the number how many lasers are in a
        cluster.
        :param size: Size of laser in a cluster.
        :return:
        """
        self._cluster_size = size

    def observation_size(self):
        """
        Get the observation vector size.
        :return: Observation vector size.
        """
        if self._cluster_size < 2:
            size = self._env.observation_size()
        else:
            size = self._env.observation_min_clustered_size(self._cluster_size)

        if self._observation_rotation_use:
            size += self._observation_rotation_size

        if self._observation_rotation_use:
            size += 2

        return size

    def visualize(self):
        """
        Visualize the current state of the simulation with gnuplot.
        :return:
        """
        end_node = self._fitness_data.get_end_node()
        self._env.visualize(end_node.x(), end_node.y(), end_node.radius())


    ########################################################
    ####         Inicio do STEP com Fronteiras          ####
    ########################################################
    def follow_path(self, frontier, map_before, robot_pose):
        start = [int(round((robot_pose[0])/self.m_resolution)),int(round((robot_pose[1])/self.m_resolution))]
        goal = [int(round((frontier[0])/self.m_resolution)),int(round((frontier[1])/self.m_resolution))]

        obst_idx = np.where(map_before == 0.0)
        obstacles = [obst_idx[0].tolist(),obst_idx[1].tolist()]
        ox = obstacles[1]
        oy = obstacles[0]


        #### ASTAR Start-END
        obst_list = []
        for i in range(len(obstacles[0])):
            obst_list.append([obstacles[0][i],obstacles[1][i]])
        obstacle = np.asarray(obst_list)
        path = searching_control(start, goal, obstacle, obstacle)


        print("Robot pose := ", robot_pose[0:2])
        # print("start := ", start)
        # print("goal := ", goal)
        # print("Path := ", path)

        # input("Wait")

        ####### Dijskstra
        # grid_size = 0.5
        # robot_radius = 1.0
        # # dijkstra = Dijkstra(np.asarray([[0.0,100.0],[0.0,100.0]]),ox, oy, grid_size, robot_radius)
        # dijkstra = Dijkstra(ox, oy, grid_size, robot_radius)

        # print("Planning")
        # print("Start :=", start)
        # # goal = [start[0], start[1] + 5]
        # print("GOAL :=", goal)
        # path = []
        # rx, ry = dijkstra.planning(start[0], start[1], goal[0], goal[1])
        # print("rx := ", rx)
        # print("ry := ", ry)
        # input("WAIT")
        # # path.append(start)
        # for j in range(len(rx)):
        #     path.append([rx[len(rx)-1-j],ry[len(rx)-1-j]])


        vec_path = np.zeros((len(path),2))
        for i in range(len(path)):
            vec_path[i,:] = list(path[i])
            vec_path[i,0] = (vec_path[i,0]*self.m_resolution + self.m_resolution/2.0)
            vec_path[i,1] = (vec_path[i,1]*self.m_resolution + self.m_resolution/2.0)


        D = 1000
        env_done = self._env.done()
        D_ant = D
        while(D > 0.5 and not env_done): 

            robot_pose = np.asarray([self._env.get_robot_pose_x(),self._env.get_robot_pose_y(),self._env.get_robot_pose_orientation()])
            D = self._distance(vec_path[i,0],vec_path[i,1],robot_pose[0],robot_pose[1])
            # print("D := ", D)

            linear, angular = self.control.control_([vec_path[i,0],vec_path[i,1]],robot_pose)

            self._env.step(linear, angular, 20)

            self.visualize()

            env_done = self._env.done()

            if((D_ant - D) >= 0.5):
                D_ant = D
                _,_= self._get_map()



        





    def detect_action(self,action):
        frontier, best = self._get_frontier()

        
        if(action ==0):
            outp = frontier[0]
        elif(action ==1):
            outp = frontier[1]
        elif(action==2):
            outp = frontier[2]
        else:
            outp = frontier[3]


        #AAAAAAAAAAAAAAAAAAAAAAAAAAAAa
        outp = best

        return outp

    def step_2(self, action):
        if (action == -1):

            mapa, _ = self._get_map()

            # cv2.imshow('Mapa',mapa)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            r_pose = np.asarray([self._env.get_robot_pose_x(),self._env.get_robot_pose_y(),self._env.get_robot_pose_orientation()])
            new_frontiers = np.zeros((4,2))

            observation = []
            observation.append(r_pose)
            observation.append(new_frontiers)
            observation.append(mapa)
            observation = np.asarray(observation,dtype=object)

            done = False
            reward = 0

        else:
            #### Select desired frontier based in discrete RNA prediction
            f_def = self.detect_action(action)

            env_robot_x = self._env.get_robot_pose_x()
            env_robot_y = self._env.get_robot_pose_y()
            env_robot_orientation = self._env.get_robot_pose_orientation()

            D = self._distance(env_robot_x, env_robot_y,f_def[0],f_def[1])

            # compute map before
            mapa_before, gmap_before = self._get_map()


            # Compute path to the desired frontier and follow
            if(action != -1):
                self.follow_path(f_def, mapa_before, np.asarray([env_robot_x,env_robot_y,env_robot_orientation]))

            # get new map
            mapa, gmap_after = self._get_map()
            map_gain = gmap_after - gmap_before

            # get new frontiers
            new_frontiers,_ = self._get_frontier()

            # compute reward
            done, reward = self.compute_reward(D, map_gain)

            # compute new robot pose
            r_pose = np.asarray([self._env.get_robot_pose_x(),self._env.get_robot_pose_y(),self._env.get_robot_pose_orientation()])

            # Compute new observation
            observation = []
            observation.append(r_pose)
            observation.append(new_frontiers)
            observation.append(mapa)
            observation = np.asarray(observation,dtype=object)

            # print(observation)
            # print(observation.shape)


        return observation, reward, done, ""


    def compute_reward(self,D, map_gain):
        done = False
        reward = 0.0

        env_done = self._env.done()
        map_reward = 0.07*float(map_gain) #/float(self.freeMap_size)
        distancy = math.log(D)

        re = distancy + map_reward

        if env_done:
            reward = -20
            done = True


        return done, reward




    @staticmethod
    def _distance(x1: float, y1: float, x2: float, y2: float) -> float:
        """
        Calculate the euler distance from to point.
        :param x1: First point x.
        :param y1: First point y.
        :param x2: Second point x.
        :param y2: Second point y.
        :return: Euler distnace from to points.
        """
        x = x1 - x2
        y = y1 - y2
        return math.sqrt(x*x + y*y)



    def reset_2(self):
        self.reset()
        self.Map = Map((self.m_height, self.m_width), self.m_resolution)

        # self.Map_Teste = localmap(self.m_height, self.m_width, self.m_resolution, [0,0])

        return self.step_2(-1)




    ########################################################
    ####            FIM do STEP com Fronteiras          ####
    ########################################################


    def step(self, linear_velocity: float, angular_velocity: float, skip_number: int = 1):
        """
        Execute a step in the simulation with the given angular and linear velocity. Return the observation, reward and
        done. If done the robot reach the goal or collided with an object.
        :param linear_velocity: Linear veloctiy of the robot.
        :param angular_velocity: Angular velocity of the robot.
        :param skip_number: Number of laserscan to skip until return.
        :return: observation, reward, done, message
        """
        self._env.step(linear_velocity, angular_velocity, skip_number)

        env_robot_x = self._env.get_robot_pose_x()
        env_robot_y = self._env.get_robot_pose_y()
        env_robot_orientation = self._env.get_robot_pose_orientation()
        env_done = self._env.done()

        reward, done = self._fitness_data.calculate_reward(env_robot_x,
                                                           env_robot_y,
                                                           env_robot_orientation,
                                                           env_done)

        if self._cluster_size < 2:
            observation = self._get_observation()
        else:
            observation = self._get_observation_min_clustered()

        # observation = self._classify(observation)  # Franzi quatsch
        # reward += 5 * observation[int(len(observation) / 2)]  # mehr Franzi quatsch

        if self._observation_rotation_use:
            not_set = True

            angle_target = self._fitness_data.angle_difference_from_robot_to_end(env_robot_x, env_robot_y, env_robot_orientation)
            angle_step_size = 2 * math.pi / self._observation_rotation_size
            angle_sum = - math.pi + angle_step_size

            for i in range(self._observation_rotation_size):
                if not_set and angle_target < angle_sum:
                    observation.append(1.0)
                    not_set = False
                else:
                    observation.append(0.0)

                angle_sum += angle_step_size

        # Including Distance and orientation to the target
        observation.append(self._fitness_data._distance_robot_to_end(env_robot_x,env_robot_y)) # ditance
        observation.append(angle_target)  # angle

        return observation, reward, done, ""

    def _classify(self, observation):
        for i in range(len(observation)):
            if observation[i] < 0.1:
                observation[i] = 1
            elif observation[i] < 0.2:
                observation[i] = 2
            elif observation[i] < 0.3:
                observation[i] = 3
            elif observation[i] < 0.4:
                observation[i] = 4
            elif observation[i] < 0.5:
                observation[i] = 5
            elif observation[i] < 0.6:
                observation[i] = 6
            elif observation[i] < 0.7:
                observation[i] = 7
            elif observation[i] < 0.8:
                observation[i] = 8
            elif observation[i] < 0.9:
                observation[i] = 9
            else:
                observation[i] = 10

        return observation


    def reset(self):
        """
        Reset the simulation. Put the robot to the (new) start position and the the (new) target position depending on
        the selected mode.
        :return:
        """

        self._fitness_data.reset()
        x, y, orientation = self._fitness_data.get_robot_start()
        self._env.set_robot_pose(x, y, orientation)
        return self.step(0.0, 0.0)

    def set_mode(self, mode, terminate_at_end=True):
        """
        Set the mode for the simulation. The mode defines the selection of the start and end node. Nodes with the same
        id are in pairs.

        *** Modes ***
        ALL_COMBINATION: Take all possible combination from start and end node. Ignore the node id.
        ALL_RANDOM: Take randomly a start and end node. Ignore the node id.
        PAIR_ALL: Take all pair combination and select from the pair a randomly start and end node.
        PAIR_RANDOM: Take randomly a pair and select from the pair a randomly start and end node.
        CHECKPOINT: Take the first start node from the lowest id and the the target node from the next higher node id.
                    When reaching the target node select the next higher id until the highest id is reached.
        :param mode: Simulation mode.
        :param terminate_at_end: Done when the target node is reached.
        :return:
        """
        self._fitness_data.set_mode(mode, terminate_at_end)