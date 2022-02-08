#!/usr/bin/env python3

import math
import numpy as np

from pysim2d import pysim2d
from .environment_fitness import FitnessData
from .environment_node import Node
# from .lidar_to_grid_map import Map, generate_ray_casting_grid_map
from .localmap import Map
from .getfrontier import getfrontier

from sklearn.cluster import DBSCAN, KMeans


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
        # self.m_morigin=[self.m_width/2.0,self.m_height/2.0]

        # self.m_height, self.m_width, self.m_resolution=12,12,0.1
        self.m_height, self.m_width, self.m_resolution=100,100,0.1
        

        

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
        mapData = self._get_map()
        
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

                        if( (len(np.where(s==205)[0]) >= 2) and (len(np.where(s1==0)[0])<=0) ):
                            x = i
                            y = j

                            front_vect.append([x,y])

        num_clusters = 4
        if(len(front_vect) > 4):
            # print(num_clusters)
            kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(front_vect)

        else: 
            size = len(front_vect)
            for i in range(num_clusters-size):
                front_vect.append(front_vect[i])

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

        return frontiers
        

        # frontiers = getfrontier(mapData)
        # return frontiers



    ########################################
    '''         Compute Map         '''
    ########################################
    def _get_map(self):

        lidar = self._get_observation_notNormalized()
        pose = [self._env.get_robot_pose_x(),self._env.get_robot_pose_y(),self._env.get_robot_pose_orientation()]

        pmap = self.Map.update_map(lidar,pose)

        return pmap


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
        # self.m=localmap(self.m_height, self.m_width, self.m_resolution,self.m_morigin)
        self.Map = Map((self.m_height, self.m_width), self.m_resolution)

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