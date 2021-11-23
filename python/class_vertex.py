########################################
# vertice da arvore
########################################
import numpy as np

########################################
########################################
class Vertex:
	########################################
	# construtor
	def __init__(self, vertice, index = None, parent = None, dim = 2):
		
		# validade do vertice
		self.valid = True
		
		# 2D or 3D
		self.dim = dim
		
		# passou um vertice (copia)
		try:
			# estados do vertice
			self.dim = vertice.dim
			self.x = vertice.x
			self.y = vertice.y
			if self.dim == 3:
				self.z = vertice.z
			self.t = vertice.t
			
			if index is None:
				# indice do vertice
				self.index = vertice.index
				# pai do vertice
				self.parent = vertice.parent
			else:
				# indice do vertice
				self.index = index
				# pai do vertice
				self.parent = parent
		
		# passou configuracao apenas	
		except AttributeError:
			# estados do vertice (tupla)
			self.x = vertice[0]
			self.y = vertice[1]
			if self.dim == 2:
				# se nao tiver tempo associado
				try:
					self.t = vertice[2]
				except:
					self.t = 0.0
			elif self.dim == 3:
				self.z = vertice[2]
				# se nao tiver tempo associado
				try:
					self.t = vertice[3]
				except:
					self.t = 0.0
			
			# indice do vertice
			self.index = index
			# pai do vertice
			self.parent = parent
			
		except:
			print("Error in vertice constructor...")
		
	########################################
	# distance rho function between two vertexes
	def dist(self, v):
		# distancia para outro vertice
		dx = self.x - v.x
		dy = self.y - v.y
		if self.dim == 2:
			dz = 0.0
		elif self.dim == 3:
			dz = self.z - v.z
			
		return np.sqrt(dx**2.0 + dy**2.0 + + dz**2.0)

	########################################
	# show vertice data
	def show(self):
		print('index: ', self.index)
		print('parent: ', self.parent)
		if self.dim == 2:
			print('loc: ', self.x, self.y, self.t)
		elif self.dim == 3:
			print('loc: ', self.x, self.y, self.z, self.t)
