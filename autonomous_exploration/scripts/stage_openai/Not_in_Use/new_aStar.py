#!/usr/bin/env python

import rospy
from nav_msgs.msg import OccupancyGrid
import cv2

import matplotlib.pyplot as plt
import numpy as np
from heapq import *
from math import pi, atan2, tan, cos, sin, sqrt, hypot, floor, ceil, log



########################################
'''           Control Class          '''
########################################
class control:
    def __init__(self):
        self.d = 0.2
        self.k = 0.3

    def control_(self,pos_curve, robot_states):

        Ux = self.k * (pos_curve[0] - robot_states[0])
        Uy = self.k * (pos_curve[1] - robot_states[1])

        return self.feedback_linearization(Ux,Uy,robot_states[2])

    def feedback_linearization(self,Ux, Uy, theta_n):

        vx = cos(theta_n) * Ux + sin(theta_n) * Uy
        w = -(sin(theta_n) * Ux)/ self.d  + (cos(theta_n) * Uy) / self.d 

        return vx, w


def callback(data):
    global Occupancy
    Occupancy = data


def A_Star(Occ_Data,start,goal):
    my_map = Occ_Data.data

    origem_map = [Occ_Data.info.origin.position.x,Occ_Data.info.origin.position.y]
    width = Occ_Data.info.width # uint
    height = Occ_Data.info.height # uint 
    resolution = Occ_Data.info.resolution # float

    rospy.loginfo(str(width) + ' ' + str(height) + ' ' + str(resolution))
    
    im = np.zeros((height,width), dtype=np.float32)

    for ii in range(0, height):
        for jj in range(0, width):
            foo = my_map[ii*width+jj]/100.0
            if foo != 0:
                im[ii][jj] = foo

    im2 = np.flipud(im)
    im2 = im2.astype(int)

    inicio = start
    final = goal
    tray = np.asarray(a_planning(im2, inicio, final))

    vec_path = np.zeros((len(tray),3))
    for i in range(len(tray)):
        s = list(tray[i])
        vec_path[i,:] = list(np.append(tray[i],0))
        vec_path[i,0] = origem_map[0] + (tray[i,0]*resolution + resolution/2.0)
        vec_path[i,1] = origem_map[1] + (tray[i,1]*resolution + resolution/2.0)
        vec_path[i,2] = 0.0

    t_x = []
    t_y = []
    t_y = vec_path[:,1]
    t_x = vec_path[:,0]


    return t_x, t_y, vec_path


class punto:

    def __init__(self, padre=None, pos=None):
        self.padre = padre
        self.pos = pos

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, sig):
        return self.pos == sig.pos


def dev_tray(punto_actual):
    tray = []
    loc = punto_actual
    while loc is not None:
        tray.append(loc.pos)
        loc = loc.padre
    return tray[::-1]  # Giramos tray -> (1234) -> (4321)


def a_planning(mapa, inicio, final):

    # Inicializamos la lista abierta/cerrada
    lista_abierta = []
    lista_cerrada = []

    # Creamos una clase asociada al punto inicial y final
    punto_inicial = punto(None, inicio)
    punto_inicial.g = 0
    punto_inicial.h = 0
    punto_inicial.f = 0
    punto_final = punto(None, final) #El punto final nos servira para determinar la H y para determinar cuando hemos llegado al destino
    punto_final.g = 0
    punto_final.h = 0
    punto_final.f = 0


    # Anadimos el punto inicial a la lista abierta
    lista_abierta.append(punto_inicial)
    
    # Condicion de parada (para evitar bucle infinito)
    iteracciones = 0
    max_iter = 15000
    

    # No permitimos movimientos en digonales
    # vecinos = ((0, -1), (0, 1), (-1, 0), (1, 0),)
    vecinos = ((0, -1), (0, 1), (-1, 0), (1, 0), (1, 1), (-1,-1), (1, -1), (-1, 1),)


    # No salimos del bucle hasta vaciar la lista
    while len(lista_abierta) > 0:
        iteracciones += 1
        
        # Obtenemos el punto actual
        punto_actual = lista_abierta[0]
        indice = 0

        for i, elemento in enumerate(lista_abierta):      #Con enumerate permitimos : [a,b,c,d] -> [1 a;2 b; 3 c;4 d]
            if elemento.f < punto_actual.f:
                punto_actual = elemento
                indice = i

        
        if iteracciones > max_iter:
            return dev_tray(punto_actual)

        # Eliminamos elemento de la lista abierta y lo anadimos a la lista cerrada
        lista_abierta.pop(indice)
        lista_cerrada.append(punto_actual)

        # En caso de haber llegado al final
        if punto_actual == punto_final:
            return dev_tray(punto_actual)

    
        hijos = []
        
        for nueva_pos in vecinos: # Recorremos los vecinos

            # Obtenemos caracterisicas del punto
            pos_punto = (punto_actual.pos[0] + nueva_pos[0], punto_actual.pos[1] + nueva_pos[1])

            # Nos aseguramos que estamos dentro del mapa
            #if pos_punto[0] > (len(mapa) - 50) or pos_punto[0] < 0 or pos_punto[1] > (len(mapa[len(mapa)-50]) -1) or pos_punto[1] < 50:
            #    continue

            # Nos aseguramos ed que el punto es valido
            if mapa[pos_punto[0]][pos_punto[1]] != 0:
                continue

            #Comprobamos si el vecino esta en la lista cerrada para evitar bucle infinito
            if punto(punto_actual, pos_punto) in lista_cerrada: 
                continue

            # Creamos nueva clase punto
            nuevo_punto = punto(punto_actual, pos_punto)

            # Anadimos
            hijos.append(nuevo_punto)

        # Recorremos a traves de los hijos
        for hijo in hijos:
            
            # Si el hijo esta en la lista cerrada
            if len([hijo_lcerrada for hijo_lcerrada in lista_cerrada if hijo_lcerrada == hijo]) > 0:
                continue

            # Calculamos las caract. del hijo
            hijo.g = punto_actual.g + 1
            hijo.h = ((hijo.pos[0] - punto_final.pos[0]) ** 2) + ((hijo.pos[1] - punto_final.pos[1]) ** 2)      
            hijo.f = hijo.g + hijo.h

            # Si el hijo esta todavia enla lista abierta
            if len([p_abierta for p_abierta in lista_abierta if hijo == p_abierta and hijo.g > p_abierta.g]) > 0:
                continue

            # Anadimos el hijo a la lista abierta
            lista_abierta.append(hijo)




def listener():
    global Occupancy

    Occupancy = OccupancyGrid()

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('grid_drawer', anonymous=True)
 
    rospy.Subscriber('map', OccupancyGrid, callback)

    rate = rospy.Rate(5)

    while not rospy.is_shutdown():
        start1 = float(input("Start point 1:"))
        start2 = float(input("Start point 2:"))
        start = (start1,start2)
        goal1 = float(input("End point 1:"))
        goal2 = float(input("End point 2:"))
        goal = (goal1,goal2)
        origem_map = [Occupancy.info.origin.position.x,Occupancy.info.origin.position.y]
        resol = Occupancy.info.resolution

        init = (int(round((start[0]-origem_map[0]-resol/2.0)/resol)),int(round((start[1]-origem_map[1]-resol/2.0)/resol)))
        end = (int(round((goal[0]-origem_map[0]-resol/2.0)/resol)),int(round((goal[1]-origem_map[1]-resol/2.0)/resol)))

        tx, ty = A_Star(Occupancy,init,end)

        print(tx)
        print("\n")
        print(ty)
        print("\n\n")


        rate.sleep()

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()




if __name__ == '__main__':
    listener()
