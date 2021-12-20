import class_vertex
import matplotlib.pyplot as plt
import numpy as np
import time
try:
	import cv2
except:
	"Nao carregou a Opencv"

########################################
# classe do mapa
########################################
class Map:
	########################################
	# construtor
	def __init__(self, xlimits, ylimits, zlimits = (0.0, 0.0), image = '', distance2obstacles = 0.0):
		
		# salva o tamanho geometrico da imagem em metros
		self.xlim = xlimits
		self.ylim = ylimits
		
		# distancia dos pontos aleatorios ate qualquer obstaculo
		self.distance2obstacles = distance2obstacles
		
		# inicializacao
		self.init2D(image)
		
		# use latex
		# try:
		# 	plt.rcParams['text.usetex'] = True
		# except:
		# 	print('Sem latex...')
		
	########################################
	# ambientes em 2D
	def init2D(self, image):
		
		# le a imagem
		I = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
		# I = cv2.resize(I, (64, 64), interpolation=cv2.INTER_NEAREST)
		
		# linhas e colunas da imagem
		self.nrow = I.shape[0]
		self.ncol = I.shape[1]

		# binariza imagem
		(thresh, I) = cv2.threshold(I, 127, 255, cv2.THRESH_BINARY)

		# inverte a imagem em y
		self.mapa = cv2.flip(I, 0)

		# mapa observado comeca todo desconhecido
		self.mapa_obs = 127*np.ones((self.nrow, self.ncol), dtype = "uint8")
		
		# parametros de conversao
		self.mx = float(self.ncol) / float(self.xlim[1] - self.xlim[0])
		self.my = float(self.nrow) / float(self.ylim[1] - self.ylim[0])
		
		# angulos de deteccao de colisao
		self.th = np.linspace(0, 2.0*np.pi, 16)
	
	########################################
	# verifica colisao com os obstaculos
	def collision(self, q, robotSize = 0.0, roubado = False):
		
		# alem do aumento dos obstaculos, soma o tamanho do robo
		distance2obstacles = robotSize + self.distance2obstacles
		
		return self.collision2D(q, distance2obstacles, roubado)
	
	########################################
	# verifica colisao com os obstaculos
	def collision2D(self, q, distance2obstacles, roubado = False):
		
		# posicao de colisao na imagem
		px, py = self.mts2px(q)
		col = int(px)
		lin = int(py)
		
		# verifica se esta dentro do ambiente
		if (lin < 0) or (lin > self.nrow):
			return True
		if (col < 0) or (col > self.ncol):
			return True
					
		# dimensoes do robo em pixels
		raio = int( np.ceil(distance2obstacles*self.mx) )
		# colisao dentro de um raio circular
		for th in self.th:
			j = col + int(raio*np.cos(th))
			i = lin + int(raio*np.sin(th))
			try:
				if not roubado:
					if self.mapa_obs.item(i, j) < 127:
						return True
				else:
					if self.mapa.item(i, j) < 127:
						return True
						
			except IndexError:
				None

		return False
		
	########################################
	# update map
	def update(self, q, laserSize):
	
		change = False
		
		# posicao de colisao na imagem		
		px, py = self.mts2px(q)
		col = int(px)
		lin = int(py)
		
		# tamanho do laser em pixels
		laser = int(laserSize*self.mx)
		
		# se algum ponto tocou obstaculos
		for i in range(lin-laser, lin+laser):
			for j in range(col-laser, col+laser):
				if np.sqrt( (i-lin)**2.0 + (j-col)**2.0 ) <= laser:
					try:
						if self.mapa_obs.item(i, j) != self.mapa.item(i, j):
							self.mapa_obs.itemset((i, j), self.mapa.item(i, j))
							change = True
					except IndexError:
						None
					
		return self.mapa_obs
		
	########################################
	# transforma pontos no mundo real para pixels na imagem
	def mts2px(self, q):
		try:
			qx = q.x
			qy = q.y
		except AttributeError:
			qx = q[0]
			qy = q[1]
		
		# conversao
		px = (qx - self.xlim[0])*self.mx
		py = self.nrow - (qy - self.ylim[0])*self.my
		
		return int(px), int(py)

	def px2mts(self,px,py):

		# conversao
		qx = float( (px/self.mx) + self.xlim[0] )
		qy = float( ((self.nrow - py)/self.my) + self.ylim[0] )
		
		return np.array([qx, qy])
		
	########################################
	# desenha a imagem distorcida em metros
	def draw(self):
		
		# plota mapa real e o mapa obsevado
		overlay = cv2.addWeighted(self.mapa_obs, 0.8, self.mapa, 0.2, 0)
		
		plt.figure(1)
		plt.imshow(overlay, cmap='gray', extent=[self.xlim[0], self.xlim[1], self.ylim[0], self.ylim[1]])

		try:
			plt.xlabel(r"$x$[m]")
			plt.ylabel(r"$y$[m]")
		except:
			plt.xlabel("x[m]")
			plt.ylabel("y[m]")
		
		plt.xticks()
		plt.yticks()
		plt.grid()
		plt.axis('equal')
		plt.xlim(self.xlim)
		plt.ylim(self.ylim)
	
	########################################
	def getMapImage(self):
		img = cv2.resize(self.mapa_obs, (64, 64), interpolation=cv2.INTER_NEAREST)
		return img

	########################################
	def getMap(self):
		n_map = np.where(self.mapa_obs==255,0,1)
		return n_map

	#######################################
	def getObstacles(self):
		m_obstacles = [np.where(self.mapa_obs==0)[0].tolist(),np.where(self.mapa_obs==0)[1].tolist()]
		return m_obstacles
	
	########################################
	# desenha a imagem distorcida em metros
	def draw_reduced(self):
		plt.figure(2)
		img = cv2.resize(self.mapa_obs, (64, 64), interpolation=cv2.INTER_NEAREST)
		plt.imshow(img, cmap='gray', extent=[self.xlim[0], self.xlim[1], self.ylim[0], self.ylim[1]])
			
	########################################
	def __exit__(self,*err):
		sim.simxFinish(-1)
		print ('Program ended')
