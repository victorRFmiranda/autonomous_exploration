################################################################################
# IMPORTS
################################################################################

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import colorsys, time

import class_robot as robot
import class_map
import class_vertex
from dijkstra import Dijkstra

################################################################################
# CONFIGS
################################################################################

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
#matplotlib.rcParams['text.usetex'] = True

font = {'family': 'sans-serif',
        'weight': 'normal',
        'size': 14}
matplotlib.rc('font', **font)

#FIG_W = 12.8
#FIG_H = FIG_W * 9/16
FIG_W = FIG_H = 10
matplotlib.rcParams['figure.figsize'] = (FIG_W, FIG_H)
matplotlib.rcParams['figure.dpi'] = 100


################################################################################
# SETUP
################################################################################

np.random.seed(30)

# globais
# DT = 1.0e-1
DT = 3
XLIMITS = (-50., 50.)
YLIMITS = (-50., 50.)
SIM_TIME = 100.0
CARSIZE = 0.1 # tamanho do robo

SALVA_IMGS = False

########################################
# cria o mapa
mapa = class_map.Map(XLIMITS, YLIMITS, image = 'imgs/muro.png', distance2obstacles = 2.0)

robo = robot.Robot(np.array([-40., -40.]), mapa)

################################################################################
# EXECUTION
################################################################################
plt.ion()

fig1 = plt.figure()
ax1 = fig1.add_subplot(111, aspect='equal')

t = 0.0
count_frame = 0

while robo.clock() < SIM_TIME:	
	# atualiza modelo
	pos, map_obs = robo.model(DT)
	
	collision = mapa.collision(robo.p,CARSIZE)
	obstacles_list = mapa.getObstacles()

	start = mapa.mts2px(robo.p)
	goal = mapa.mts2px(np.asarray([-38.0,-38.0]))

	print("Start = ", start)
	print("Goal = ", goal)

	dijkstra = Dijkstra(np.asarray([[XLIMITS[0],XLIMITS[1]],[YLIMITS[0],YLIMITS[1]]]),obstacles_list[0], obstacles_list[1], 1.0, CARSIZE)
	rx, ry = dijkstra.planning(start[0], start[1], goal[0], goal[1])
	path = []
	for j in range(len(rx)):
		path.append([rx[len(rx)-1-j],ry[len(rx)-1-j]])

	print(path)

	

	# start = [int(round((self.robot_pose[0]-self.origem_map[0]-self.resol/2.0)/self.resol)),int(round((self.robot_pose[1]-self.origem_map[1]-self.resol/2.0)/self.resol))]
	# goal = [int(round((point[0]-self.origem_map[0]-self.resol/2.0)/self.resol)),int(round((point[1]-self.origem_map[1]-self.resol/2.0)/self.resol))]
	# print(pos)

	# print(obstacles_list)

	# desenha de vezes em quando
	if (t % 1) < DT:
		ax1.cla()
		if not SALVA_IMGS:
			plt.title('Time: %.1lf s' % t)
		
		# desenha o mapa
		mapa.draw()
		
		# desenha os carros
		robo.draw()
		
		# desenha
		# salva para animacao
		if SALVA_IMGS:
			fig1.savefig("pngs/exp1/%03d.png" % count_frame, bbox_inches='tight')
			fig1.savefig("pdfs/exp1/%3.2f.pdf" % t, bbox_inches='tight')
			count_frame = count_frame + 1
			
		plt.pause(0.1)
	
	t = t + DT
	# end while
	
################################################################################
print("Terminou...")
#raise SystemExit()

plt.ioff()
plt.show()

