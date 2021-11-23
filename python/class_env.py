# -*- coding: utf-8 -*-
import numpy as np
from gym import spaces
import gym
gym.logger.set_level(40)
import matplotlib.pyplot as plt

import class_map
import class_robot as robot

########################################
# GLOBAIS
########################################
DT = 1.0e-1
XLIMITS = (-50., 50.)
YLIMITS = (-50., 50.)

SALVA_IMGS = False

########################################
# Enviroment
########################################
class Env(gym.Env):
	########################################
	# construtor
	########################################
	def __init__(self, args):
		
		self.action_space = args.num_actions		# num frontiers (fixed)
		self.observation_space = np.asarray([])
		self.max_actions = args.MAX_STEPS			# num actions for epoch (how many times check all frontiers)
		self.num_initstates = args.NUM_EPISODES 	# num start positions before change the map
		self.maps = args.maps_gt					# vector with name of stage maps for training
		self.map_count = 0

		#self.init_pose = [-40.0, -40.0, 0.0]		# x, y, theta  -- Came from a parameters ?
		#self.robot_pose = [0.0, 0.0, 0.0]			# x, y, theta
		#self.f_points = []
		#self.step_count = 0
		#self.freeMap_size = 0
		
		
	########################################
	# seed
	########################################
	def seed(self, rnd_seed = None):
		np.random.seed(rnd_seed)
		return [rnd_seed]
		
	########################################
	# reset
	########################################
	def reset(self):
		
		# cria o mapa
		self.mapa = class_map.Map(XLIMITS, YLIMITS, image = 'imgs/cave.png')

		# cria robo
		self.robot = robot.Robot(np.array([-40.0, -40.0]), self.mapa)
		
		# cria fronteiras
		self.frontier = []
		for i in range(4):
			self.frontier.append([i,2*i])
		self.frontier = np.array(self.frontier)
		
		
		# estado
		self.state = np.asarray([self.robot.getPose(), self.frontier, self.mapa.getMap()])
		
		return self.state
		
	########################################
	# step -> new_observation, reward, done, info = env.step(action)
	def step(self, action):
		
		# atualiza modelo
		carro.model(DT)
		
		#new_observation, reward, done, info
		return self.state, reward, done, {}
					
	########################################
	# desenha
	def render(self):
		
		# desenha de vezes em quando
		if (t % .5) < DT:
			fig1 = plt.figure(1)
			fig1.clf()
			ax1 = fig1.add_subplot(111, aspect='equal')
			plt.title('Time: %.1lf s' % t)
			
			# desenha o mapa
			mapa.draw()
			
			# desenha os carros
			carro.draw()
			
			# salva para animacao
			if SALVA_IMGS:
				fig1.savefig("pngs/%03d.png" % count_frame, bbox_inches='tight')
				count_frame = count_frame + 1
		
			mapa.draw_reduced()
			
			plt.pause(0.1)
		
		
		
	########################################
	# fecha ambiente
	def close(self):
		
		plt.ioff()
		
	########################################
	# termina a classe
	def __del__(self):
		None

