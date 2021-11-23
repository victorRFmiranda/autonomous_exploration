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
DT = 1.0e-1
XLIMITS = (-50., 50.)
YLIMITS = (-50., 50.)

SALVA_IMGS = False

########################################
# cria o mapa
mapa = class_map.Map(XLIMITS, YLIMITS, image = 'imgs/cave.png', distance2obstacles = 2.0)

robo = robot.Robot(np.array([-40., -40.]), mapa)

################################################################################
# EXECUTION
################################################################################
plt.ion()

fig1 = plt.figure()
ax1 = fig1.add_subplot(111, aspect='equal')

t = 0.0
count_frame = 0

while robo.clock() < 20.0:	
	# atualiza modelo
	robo.model(DT)

	# desenha de vezes em quando
	if (t % .5) < DT:
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

