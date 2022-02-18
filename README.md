# Deep Reinforcement Learning for Autnomous exploration

DRL training algorithm for selection of the best exploration frontier in under discover environment.

## Stand-aloe (2D_SIM)

### Dependencies
```
$ sudo apt install gnuplot
```

### Main CODE
Actor-Critic method for selection of the best exploration frontier and reach this point using a path planning algorithm:
```
$ python3 AC_TD_Foward.py
```
Actor-Critic method considering reach a point in an environment avoiding of obstacles:
```
$ python3 AC_TD_FW_navigation.py
```
Asynchronous Advantage Actor Critic (A3C) method considering reach a point in an environment avoiding of obstacles, training on different maps in parallel:
```
$ python3 A3C.py
```
### Auxiliary codes
Simulator Source Codes are in the following folder `2D_SIM/Simulation2d`

The simulation python library are in `2D_SIM/pysim2d`

The environment code for DRL training, the frontier detection code, occupancy grid map generation code, and others are in `2D_SIM/environment`


## ROS (autonomous_exploration)

### ROS Dependencies
```
$ sudo apt install ros-"your ros distro"-gmapping
$ sudo apt install ros-"your ros distro"-map-server
```

### Python Dependencies
```
Numpy
Pytorch
Scikit-learn
Seaborn
Matplotlib
Gym
```


### Building Package

Install this package and the modified version of ROS Stage simulator on your catking workspace:

```
$ cd ~/catkin_ws/src/
$ git clone https://github.com/victorRFmiranda/autonomous_exploration.git
$ git clone https://github.com/victorRFmiranda/stage_ros.git
$ cd ..
$ catkin_make ## or catkin build
```

### Main CODE

The main code is located in:
`/scripts/stage_openai/AC_TD_lambda_forward.py`

Other auxiliary codes are:

Path Planner: `/scripts/stage_openai/a_star.py`

DRL Configuration File: `/scripts/stage_openai/config.py`

Training Environment: `/scripts/stage_openai/ros_stage_env.py`

Frontier Detection Algorithm: `/scripts/frontier/frontier_lidar.py`

### How to run

In order to start the training, run the following codes in two different terminal windows:

First:
```
$ roscore
```

Second:
```
$ rosrun autonomous_exploration AC_TD_lambda_forward.py
```
