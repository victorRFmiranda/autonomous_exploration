#!/usr/bin/env python

import math
import numpy as np
import cv2
from collections import deque

from pysim2d import pysim2d
from .environment_fitness import FitnessData
from .environment_node import Node
# from .lidar_to_grid_map import Map, generate_ray_casting_grid_map
# from .localmap import Map
# from .create_map import localmap
# from .getfrontier import getfrontier
# from .dijkstra import Dijkstra
# from .astar_start_end import searching_control
# from sklearn.cluster import DBSCAN, KMeans

# from .grid_map import *
# from .utils import *

import warnings

import numpy as np

warnings.filterwarnings('ignore')


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

        self.m_height, self.m_width, self.m_resolution=100,100,0.5
        self.P_prior = 0.5
        self.P_occ = 0.9
        self.P_free = 0.3
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

    def _get_observation_intervals(self):
        size = self._env.observation_size()
        beams = []
        observation = []

        for i in range(size):
            beams.append(self._env.observation_at(i))

        for j in range(0,size,12):
            observation.append(min(beams[j:j+11]))

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

        # consider relative position to target and robot velocity
        size += 4

        return size

    def visualize(self):
        """
        Visualize the current state of the simulation with gnuplot.
        :return:
        """
        end_node = self._fitness_data.get_end_node()
        self._env.visualize(end_node.x(), end_node.y(), end_node.radius())



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

        # observation = self._get_observation()
        observation = self._get_observation_intervals()

        reward, done = self._fitness_data.calculate_reward(env_robot_x,
                                                           env_robot_y,
                                                           env_robot_orientation,
                                                           env_done, 
                                                           linear_velocity, angular_velocity, observation)


        angle_target = self._fitness_data.angle_difference_from_robot_to_end(env_robot_x, env_robot_y, env_robot_orientation)


        # Including polar coord to target
        # AQUUUUUUUUUUUUUUUUUUUUUUIIIIIIIIIIIIIIIII
        observation.append(self._fitness_data._distance_robot_to_end(env_robot_x,env_robot_y)) # ditance
        observation.append(angle_target)  # angle

        return np.asarray(observation), reward, done, ""

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