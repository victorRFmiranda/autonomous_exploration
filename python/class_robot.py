########################################
# robot
########################################
import numpy as np
from class_vertex import *
import colorsys
import matplotlib.pyplot as plt

CARSIZE = 0.1 # tamanho do robo
LASER = 10.0 # raio do laser
VELMAX = 10.0
CONTROLGAIN = 5.0
        
########################################
########################################
class Robot:
	
	########################################
	# construtor
	def __init__(self, p, mapa):
		
		# condicoes iniciais
		self.p = np.array(p)
		self.v = np.zeros(2)
		self.t = 0.0 # timer interno
		
		# seta cor padrao
		self.color = np.array(colorsys.hsv_to_rgb(0, .8, .8))
		
		# atribui mapa
		self.setMap(mapa)
		
		# trajetoria de referencia para o controlador
		self.trajref = []
		self.ref = Vertex(self.p, index = 0, parent = [])
		
		# salva condicoes iniciais
		self.saveTraj()
	
	########################################
	# acao de controle para seguir o melhor caminho
	def control(self):
		
		########################
		# define a referencia
		########################
		# pega a posicao atual do veiculo
		position = Vertex(self.p, index = 0, parent = [])
		
		# distancia maxima ate a referencia
		the_dist = 0.5
		
		# calcula a proxima referencia livre de colisao
		achou = False
		for v in reversed(self.trajref):
			if v.dist(position) > the_dist:
				if v.index >= self.ref.index:
					if not self.tree.collisionBetweenVertexes(v, position):
						self.ref = v
						achou = True
						break
		
		# se nao achou, reseta o planejador
		if not achou:
			self.ref.index = 0
			self.resetPlanner()
			return np.zeros(2)
		
		########################
		# joga ref para o controlador
		########################
		# calcula o angulo de direcao do controle
		dx = self.ref.x - position.x
		dy = self.ref.y - position.y
		th = np.arctan2(dy, dx)
		# calcula velocidade de controle
		d = np.linalg.norm(np.array([dx, dy]))
		
		# comando de velocidade
		u = CONTROLGAIN*d*np.array([np.cos(th), np.sin(th)])
		return self.satSpeed(u)
		
	########################################
	# satura velocidade do robo
	def satSpeed(self, u):
		# aplica limitacao de velocidade
		nu = np.linalg.norm(u)
		if nu > VELMAX:
			u = VELMAX*(u/nu)
		return u
		
	########################################
	# atualiza o modelo do robo
	def model(self, dt, potentialField = False):
		
		# integra o tempo
		self.t += dt
		
		#u = np.zeros(2)
		u = np.ones(2)
		u = self.satSpeed(u)

		# update mapa
		self.mapUpdate()

		##################
		# aplica controlador ao robo
		self.v = u
		self.p = self.p + self.v*dt	
		
		# salva trajetoria
		self.saveTraj()
					
	########################################
	# update do mapa
	def mapUpdate(self):
		
		# se informacao do laser altera o mapa
		return self.mapa.update(self.p, LASER)
			
	########################################
	# informa ao robo o mapa do ambiente
	def setMap(self, mapa):
		self.mapa = mapa
	
	########################################
	def getPose(self):
		return np.array([self.p[0], self.p[1], 0.0]) #self.p
		
	########################################
	# get timer
	def clock(self):
		return self.t

	########################################
	# salva os estados do sistema
	def saveTraj(self):
		try:
			# se ja iniciou as trajetorias
			self.traj.append((self.t, self.p, self.v))
		except:
			# se for a primeira vez
			self.traj = [(self.t, self.p, self.v)]
			
	########################################
	# desenha o robo e outras informacoes
	def draw(self):
		
		# desenha o carro
		plt.plot(self.p[0], self.p[1], 'o', color = self.color, linewidth = 1, markersize = 10)
		
		# desenha laser
		circle = plt.Circle(xy = [self.p[0], self.p[1]], radius = LASER, fc = self.color, ec = 'black', alpha = 0.01)
		plt.gca().add_patch(circle)
		
		# desenha a trajetoria do carro
		px = [traj[1][0] for traj in self.traj]
		py = [traj[1][1] for traj in self.traj]
		plt.plot(px, py, '--', color = self.color, linewidth = 2)
