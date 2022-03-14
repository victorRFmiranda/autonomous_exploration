#!/usr/bin/env python

import math
import time
import rospy
from enum import Enum
from itertools import groupby


class DirectionType(Enum):
    LEFT = 0
    RIGHT = 1


class StateType(Enum):
    FOLLOW_VECTOR_FIELD = 0
    FOLLOW_OBSTACLE = 1


class VecFieldController:
    """
    Doc here #todo
    """

    def __init__(self, v_r, k_f, d_feedback, epsilon, switch_dist, is_forward_motion_flag, flag_follow_obstacle):
        # base variables
        self.pos = [0, 0, 0]
        self.rpy = [0, 0, 0]
        self.traj = []
        self.state_k = 0
        self.state_k_delta = 10

        # potential field repulsion
        self.epsilon = epsilon
        self.switch_dist = switch_dist

        # controller constants
        self.v_r = v_r
        self.k_f = k_f
        self.d_feedback = d_feedback

        # flags
        self.is_forward_motion_flag = is_forward_motion_flag
        self.flag_follow_obstacle = flag_follow_obstacle
        self.closed_path_flag = False

        # obstacle avoidance point information
        self.delta_m = 1000.0  # minimum distance
        self.phi_m = 0.0  # angle minimum distance (body frame)

        # initial controller states
        self.robot_action_state = StateType.FOLLOW_VECTOR_FIELD
        self.obstacle_dir = DirectionType.RIGHT

    def set_pos(self, pos, rpy):
        self.pos = pos
        self.rpy = rpy

    def set_obstacle_point(self, point):
        """Callback to get the closest point obtained with the lidar
        used for obstacle avoidance
        :param point: 2D point list or tuple (x, y)
        """
        x, y = point
        x = x-self.d_feedback

        self.phi_m = math.atan2(y, x)
        self.delta_m = math.sqrt(x * x + y * y)
        print("self.phi_m: ", self.phi_m)
        print("self.delta_m: ", self.delta_m)

    def set_trajectory(self, traj, insert_n_points, filter_path_n_average, closed_path_flag):
        """Callback to obtain the trajectory to be followed by the robot
        :param data: trajectory ROS message
        """
        self.reset()

        # remove consecutive points and check for repeated points at the start and tail
        traj = [x[0] for x in groupby(traj)]
        if len(traj) > 1:
            if traj[0] == traj[-1]:
                traj.pop(-1)

        self.traj = traj
        self.closed_path_flag = closed_path_flag

        # Insert points on the path
        if insert_n_points > 0:
            self.traj = self.insert_points(self.traj, insert_n_points, closed_path_flag)

        # Filter the points (average filter)
        if filter_path_n_average > 0:
            self.traj = self.filter_path(self.traj, filter_path_n_average, closed_path_flag)

        # Update the closest index - index to the closest point in the curve
        self.state_k = 0
        d = float("inf")
        for k in range(len(self.traj)):
            d_temp = math.sqrt((self.pos[0] - self.traj[k][0]) ** 2 + (self.pos[1] - self.traj[k][1]) ** 2)
            if d_temp < d:
                self.state_k = k
                d = d_temp

        rospy.loginfo("New path received by the controller (%d points)", len(self.traj))

    def set_forward_direction(self, is_forward):
        self.is_forward_motion_flag = is_forward

    def reset(self):
        self.robot_action_state = StateType.FOLLOW_VECTOR_FIELD
        self.obstacle_dir = DirectionType.RIGHT
        #self.pos = [0, 0, 0]
        #self.rpy = [0, 0, 0]
        self.traj = []
        self.state_k = 0
        self.state_k_delta = 10

    def is_ready(self):
        return True if self.traj and len(self.traj) > 0 else False

    def get_traj(self):
        return self.traj

    def vec_field_path(self):
        """Compute the vector field that will guide the robot through a path
        :return:
            Vx, Vy, reached_endpoint, (reached_percentage if not reached_endpoint else 100)
        """
        x, y, z = self.pos
        local_traj = self.traj
        size_traj = len(local_traj)
        reached_endpoint = False
        reached_percentage = 0.0

        # Compute the closest ponit on the curve
        # Consider only the points in the vicinity of the current closest point (robustness)
        k_vec = [self.state_k - self.state_k_delta + i for i in range(self.state_k_delta)]
        k_vec.append(self.state_k)
        k_vec = k_vec + [self.state_k + 1 + i for i in range(self.state_k_delta)]
        for k in range(len(k_vec)):
            if k_vec[k] < 0:
                k_vec[k] = k_vec[k] + size_traj
            if k_vec[k] >= size_traj:
                k_vec[k] = k_vec[k] - size_traj

        # iterate over the k_vec indices to get the closest point
        D = float("inf")
        k_min = size_traj
        for k in k_vec:
            D_temp = math.sqrt((x - local_traj[k][0]) ** 2 + (y - local_traj[k][1]) ** 2)
            if D_temp < D:
                k_min = k
                D = D_temp
        self.state_k = k_min  # index of the closest point

        if self.state_k > 0.0:
            reached_percentage = (self.state_k * 100) / float(size_traj)
        else:
            reached_percentage = 0.0

        # compute the distance vector
        D_vec = [x - local_traj[k_min][0], y - local_traj[k_min][1]]
        # compute the gradient of the distance Function
        grad_D = [D_vec[0] / (D + 0.000001), D_vec[1] / (D + 0.000001)]

        # compute neighbors of k_min
        k1 = k_min - 1  # previous index
        k2 = k_min + 1  # next index
        if self.closed_path_flag:
            # consider that the first point and the last are neighbors
            if k1 == -1:
                k1 = size_traj - 1
            if k2 == size_traj:
                k2 = 0
        else:
            # consider that the first point and the last are distant apart
            if k1 == -1:
                k1 = 0
            if k2 == size_traj:
                k2 = size_traj - 1

        # numerically compute the tangent vector and normalize it
        T = [local_traj[k2][0] - local_traj[k1][0], local_traj[k2][1] - local_traj[k1][1]]
        norm_T = math.sqrt(T[0] ** 2 + T[1] ** 2) + 0.000001
        
        T = [T[0] / norm_T, T[1] / norm_T]

        # lyapunov Function
        P = 0.5 * (D ** 2)
        # Gain functions
        G = -(2 / math.pi) * math.atan(self.k_f * math.sqrt(P))  # convergence
        H = math.sqrt(1 - G ** 2)  # circulation

        # compute the field's components
        Vx = self.v_r * (G * grad_D[0] + H * T[0])
        Vy = self.v_r * (G * grad_D[1] + H * T[1])

        # Stop the robot if the it reached the end of a open path
        if not self.closed_path_flag:
            if k_min == size_traj - 1:
                rospy.logwarn("CHECK THIS: k_min:%s size_traj-1:%s self.pos:%s local_traj[k_min]:%s", 
                    k_min, size_traj - 1, self.pos, local_traj[k_min])

                Vx = 0
                Vy = 0
                reached_endpoint = True
                self.reset()

        return Vx, Vy, reached_endpoint, (reached_percentage if not reached_endpoint else 100)

    def vec_field_point(self):
        """Compute the vector field that will guide the robot towards a point
        :return:
            Vx, Vy, reached_endpoint, (reached_percentage if not reached_endpoint else 100)
        """
        x, y, z = self.pos
        local_traj = self.traj
        reached_endpoint = False
        reached_percentage = 0.0

        # compute the distance vector
        D_vec = [x - local_traj[0][0], y - local_traj[0][1]]
        D = math.sqrt(D_vec[0] ** 2 + D_vec[1] ** 2)
        grad_D = [D_vec[0] / (D + 0.000001), D_vec[1] / (D + 0.000001)]

        G = -(2 / math.pi) * math.atan(self.k_f * D)  # convergence

        # compute the field's components
        Vx = self.v_r * (G * grad_D[0])
        Vy = self.v_r * (G * grad_D[1])

        if D < 0.1:
            reached_endpoint = True
            reached_percentage = 100

        return Vx, Vy, reached_endpoint, reached_percentage

    def compute_command(self, obstacle_dir):
        """Function to compute the control law
        :return:
        """
        G = (2 / math.pi) * math.atan(self.k_f * (self.delta_m - self.epsilon))

        if obstacle_dir == DirectionType.LEFT:
            H = math.sqrt(1 - G * G)
        else:
            H = -math.sqrt(1 - G * G)

        v = self.v_r * (math.cos(self.phi_m) * G - math.sin(self.phi_m) * H)
        omega = self.v_r * (math.sin(self.phi_m) * G / self.d_feedback + math.cos(self.phi_m) * H / self.d_feedback)

        return v, omega

    @staticmethod
    def insert_points(original_traj, qty_to_insert, closed_path_flag):
        """Insert points in the received path
        :param original_traj: original trajectory
        :param qty_to_insert: number of points to insert between two pair of points
        :param closed_path_flag: boolean to define if its going to be
                                 insertion of points between last and first
        :return: a new trajectory with the interpolated paths
        """
        new_traj = []
        traj_size = len(original_traj)

        if closed_path_flag:
            # Insert points between last and first
            for i in range(traj_size):
                new_traj.append(original_traj[i])

                iM = (i + 1) % traj_size
                for j in range(1, qty_to_insert + 1):
                    alpha = j / (qty_to_insert + 1.0)
                    px = (1 - alpha) * original_traj[i][0] + alpha * original_traj[iM][0]
                    py = (1 - alpha) * original_traj[i][1] + alpha * original_traj[iM][1]
                    new_traj.append((px, py))

        else:
            # Do not insert points between last and first
            for i in range(traj_size - 1):
                new_traj.append(original_traj[i])
                iM = i + 1
                for j in range(1, qty_to_insert + 1):
                    alpha = j / (qty_to_insert + 1.0)
                    px = (1 - alpha) * original_traj[i][0] + alpha * original_traj[iM][0]
                    py = (1 - alpha) * original_traj[i][1] + alpha * original_traj[iM][1]
                    new_traj.append((px, py))

        return new_traj

    @staticmethod
    def filter_path(original_traj, filter_path_n_average, closed_path_flag):
        """Filter the path using an average filter
        :param original_traj: original trajectory
        :param filter_path_n_average:
        :param closed_path_flag: boolean to define if its going to be
                                 insertion of points between last and first
        :return: a filtered list of points
        """
        size_original_traj = len(original_traj)

        if filter_path_n_average > size_original_traj:
            rospy.logwarn("Parameter 'filter_path_n_average' seems to be too high! (%d)", filter_path_n_average)

        # Force the a odd number, for symmetry
        if filter_path_n_average % 2 == 0:
            filter_path_n_average = filter_path_n_average + 1
        half = int((filter_path_n_average - 1.0) / 2.0)

        # Compute a list of shifts to further obtain the neighbor points
        ids = []
        for i in range(filter_path_n_average):
            ids.append(i - half)

        # Initialize a new list with zeros
        new_traj = []
        for i in range(size_original_traj):
            new_traj.append((0.0, 0.0))

        # For each point in the path compute the average of the point and its neighbors
        if closed_path_flag:
            # Consider a "circular" filter
            for i in range(size_original_traj):
                for j in ids:
                    k = (i + j) % size_original_traj
                    px = new_traj[i][0] + original_traj[k][0] * float(1.0 / filter_path_n_average)
                    py = new_traj[i][1] + original_traj[k][1] * float(1.0 / filter_path_n_average)
                    new_traj[i] = (px, py)
        else:
            # Consider a standard filter
            for i in range(size_original_traj):
                count = 0
                for j in ids:
                    k = (i + j)
                    # Decrease the number of elements in the extremities
                    if 0 <= k < size_original_traj:
                        count = count + 1
                        px = new_traj[i][0] + original_traj[k][0]
                        py = new_traj[i][1] + original_traj[k][1]
                        new_traj[i] = (px, py)

                avg_px = new_traj[i][0] / float(count)
                avg_py = new_traj[i][1] / float(count)
                new_traj[i] = (avg_px, avg_py)

        return new_traj

    def feedback_linearization(self, Ux, Uy):
        """Function feedback linearization
        :param Ux:
        :param Uy
        :return:
        """
        psi = self.rpy[2]  # yaw angle

        # compute forward velocity and angular velocity
        VX = math.cos(psi) * Ux + math.sin(psi) * Uy
        WZ = (-math.sin(psi) / self.d_feedback) * Ux + (math.cos(psi) / self.d_feedback) * Uy

        return VX, WZ

    def run_one_cycle(self):
        """Execute one cycle of the controller loop
        :return:
        """

        local_traj = self.traj
        size_traj = len(local_traj)
        if(size_traj>1):
            Vx_ref, Vy_ref, reached_endpoint, reached_percentage = self.vec_field_path()
        elif(size_traj==1):
            Vx_ref, Vy_ref, reached_endpoint, reached_percentage = self.vec_field_point()
        else:
            return

        #rospy.loginfo("reached_endpoint:%s, reached_percentage:%d", reached_endpoint, reached_percentage)
        V_forward, w_z = 0.0, 0.0

        if self.flag_follow_obstacle:
            # compute vector that rules the switch of states
            b_x = self.delta_m * math.cos(self.phi_m + self.rpy[2])
            b_y = self.delta_m * math.sin(self.phi_m + self.rpy[2])

            #TEMP......
            #self.robot_action_state = StateType.FOLLOW_OBSTACLE
            #self.obstacle_dir = DirectionType.RIGHT
            #TEMP......

            if self.robot_action_state == StateType.FOLLOW_VECTOR_FIELD:
                V_forward, w_z = self.feedback_linearization(Vx_ref, Vy_ref)
                if self.delta_m < self.switch_dist:
                    if Vx_ref * b_x + Vy_ref * b_y > 0:
                        self.robot_action_state = StateType.FOLLOW_OBSTACLE
                        cross_product = Vx_ref * b_y - Vy_ref * b_x
                        if cross_product > 0:
                            self.obstacle_dir = DirectionType.RIGHT
                            #rospy.loginfo("Following obstacle right")
                        else:
                            self.obstacle_dir = DirectionType.LEFT
                            #rospy.loginfo("Following obstacle left")

            elif self.robot_action_state == StateType.FOLLOW_OBSTACLE:
                V_forward, w_z = self.compute_command(self.obstacle_dir)
                if Vx_ref * b_x + Vy_ref * b_y < 0.1:
                    self.robot_action_state = StateType.FOLLOW_VECTOR_FIELD
                    #rospy.loginfo("Following vector field ...")

        else:
            # if the robot must only follow the vector field compute a command of velocity
            V_forward, w_z = self.feedback_linearization(Vx_ref, Vy_ref)

        linear_vel_x = V_forward
        if not self.is_forward_motion_flag:
            # linear_vel_x = -V_forward
            w_z = -w_z

        angular_vel_z = w_z
        return linear_vel_x, angular_vel_z, Vx_ref, Vy_ref, reached_endpoint, reached_percentage
