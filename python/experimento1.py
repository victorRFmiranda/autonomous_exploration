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
from a_star import Astar

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

# atualiza modelo
pos, map_obs = robo.model(DT,np.zeros(2))

count = 0
s = [[-35,-35],[-30,-30],[-25,-25],[-20,-20],[-15,-15]]
while robo.clock() < SIM_TIME:	
		
	collision = mapa.collision(robo.p,CARSIZE)
	obstacles_list = mapa.getObstacles()

	occ_grid = mapa.getMap()
	obst_idx = np.where(occ_grid == 1)
	obstacles = [obst_idx[0].tolist(),obst_idx[1].tolist()]
	ox = obstacles[1]
	oy = obstacles[0]
	

	start = mapa.mts2px(robo.p)
	# goal = mapa.mts2px(np.asarray([-38.0,-38.0]))
	goal = mapa.mts2px(np.asarray(s[count]))

	print("Start = ", start)
	print("Goal = ", goal)

	# path = Astar(occ_grid,start,goal)
	# print(path)


	dijkstra = Dijkstra(np.asarray([[XLIMITS[0],XLIMITS[1]],[YLIMITS[0],YLIMITS[1]]]),ox, oy, 1.0, CARSIZE)
	print("Planning")
	rx, ry = dijkstra.planning(start[0], start[1], goal[0], goal[1])
	print("Finish")
	path = []
	for j in range(len(rx)):
		path.append([rx[len(rx)-1-j],ry[len(rx)-1-j]])

	# print(len(path))

	n_path = np.zeros((len(path),2))
	for k in range(len(path)):
		n_path[k] = mapa.px2mts(path[k][0],path[k][1])

		U = robo.control(n_path[k])
		print(U)
		# print("AAA")
		# atualiza modelo
		pos, map_obs = robo.model(DT,U)

	count += 1

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

