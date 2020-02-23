import math
import random
import numpy as np
from tkinter import *
import gym
from gym import spaces
from gym.utils import seeding
import time
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

from .CR import CR as rb


def goal_distance(goal_a, goal_b):
	return np.linalg.norm(goal_a-goal_b,axis=-1)

def string_parser(string):
    string = string.replace('[','')
    string = string.replace(']','')
    x,y,z = string.split(',')
    x=np.interp(float(x), [250,550], [0,10])
    y=np.interp(float(y), [-220,220], [0,10])
    z=np.interp(float(z), [60,390], [0,10])
    
    return x, y, z
def funcionlimites(positionx,positiony,positionz):
    x0=np.interp(290, [250,550], [0,10])
    y0=np.interp(-150, [-220,220], [0,10])
    z0=np.interp(60, [60,390], [0,10])
    x1=np.interp(527, [250,550], [0,10])
    y1=np.interp(102, [-220,220], [0,10])
    z1=np.interp(240, [60,390], [0,10])


    if (positionx>x0 and positionx<x1) and (positiony>y0 and positiony<y1) and (positionz>z0 and positionz<z1):
      return True
    return False

#Entorno que funciona como un entorno de gym para poder interactuar con un agente desarrolado 
#por OPENAI, este entorno esta preparado para funcionar con HER+DDPG 
class RobotStudio(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self):
        self.min_x_position = 0						#Parámetros para limitar los movimientos en cada eje
        self.max_x_position = 10
        self.min_y_position = 0
        self.max_y_position = 10
        self.min_z_position = 0
        self.max_z_position = 10
        self.speed = 0.8
        self.goal_x_position = random.uniform(self.min_x_position,self.max_x_position)				#Generamos una posición aleatoria para el goal a alcanzar
        self.goal_y_position = random.uniform(self.min_y_position,self.max_y_position)
        self.goal_z_position = random.uniform(0,3.5)
        self.threshold=0.1
        self.x=random.uniform(self.min_x_position, self.max_x_position)								#Generamos una posición inicial aleatoria
        self.y=random.uniform(self.min_y_position, self.max_y_position)
        self.z=random.uniform(self.min_y_position, self.max_y_position)
        self.z=0
        self.envio=1


        self.low = np.array([self.min_x_position, self.min_y_position,self.min_z_position])			#Nos permite limitar los movimientos según las variables creadas anteriormente
        self.high = np.array([self.max_x_position, self.max_y_position,self.max_z_position])
        self.low_action = np.array([-self.speed, -self.speed,-self.speed])							#Nos permite limitar la máxima distancia que puede recorrer en cada step
        self.high_action = np.array([self.speed, self.speed,self.speed])
        self.viewer = None

        #self.action_space = spaces.Discrete(4)
        self.action_space = spaces.Box(self.low_action, self.high_action, dtype=np.float32)			#Generamos el espacio de acción (las acciones que puede realizar)
        self.observation_space = spaces.Dict(dict(																							#Generamos el espacio de observaciones (los estados que puede tomar)
            desired_goal=spaces.Box(self.low, self.high, dtype='float32'),
            achieved_goal=spaces.Box(self.low, self.high, dtype='float32'),
            observation=spaces.Box(self.low, self.high, dtype='float32')))

        self.seed()
        fig = plt.figure()																																			
        self.ax = Axes3D(fig)
        self.ax.set_xlim3d(self.min_x_position,self.max_x_position)
        self.ax.set_ylim3d(self.min_y_position,self.max_y_position)
        self.ax.set_zlim3d(self.min_z_position,self.max_z_position)

        self.sock=rb.RS_serial()        #Función para conectar el python al robotstudio
        self.sock.RS_connect()

        #ax.axis([self.min_x_position, self.max_x_position, self.min_y_position, self.max_y_position,self.min_z_position, self.max_z_position])


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def compute_reward(self, achieved_goal,goal,info):																	#Función para generar los rewards
        #d=math.sqrt((achieved_goal[0]-goal[0])**2+(achieved_goal[1]-goal[1])**2)
        d=goal_distance(achieved_goal,goal)
        reward = (d>self.threshold)
        return -np.float32(reward)

    def _is_success(self,achieved_goal,desired_goal):																		#Función para decirnos el porcentaje de éxito despues de cada episodio
        d=goal_distance(achieved_goal,desired_goal)
        reward = (d<self.threshold)
        return np.float32(reward)

    def step(self, action):
    
        
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action)) 
        
        positionx, positiony, positionz = self.state
																	#Generamos el estado según las posiciones en x, y y z
        '''if action==0:
            positiony-=self.speed
        if action==2:
            positiony+=self.speed
        if action==1:
            positionx+=self.speed
        if action==3:
            positionx-=self.speed
        '''
        
        action[1] = np.clip(action[1], 																						#Para hacer el movimiento más fluido, permitimos al agente moverse según una esfera.
                    -math.sqrt(abs((self.speed)**2-(action[0])**2)),														#Para ello, limitamos el movimiento en y (action[1]) a una circunferencia con 
                     math.sqrt(abs((self.speed)**2-(action[0])**2)))														#respecto al movimiento en x (action[0]), y el movimiento en z (action [2]) con respecto
                                                                                                                                                                            #a la función de la esfera, usando como parámetros los movimientos en x y en y
        action[2] = np.clip(action[2], 
                    -math.sqrt(abs((self.speed)**2-(action[0])**2-(action[1])**2)),
                     math.sqrt(abs((self.speed)**2-(action[0])**2-(action[1])**2)))

        
        if funcionlimites(positionx+action[0],positiony+action[1],positionz+action[2]): 
        	action[0]=0
        	action[1]=0
        	action[2]=0
        '''
        positionx += action[0]
        positiony += action[1]
        positionz +=action[2]
        positionx = np.clip(positionx, self.min_x_position, self.max_x_position)
        positiony = np.clip(positiony, self.min_y_position, self.max_y_position)
        positionz = np.clip(positionz, self.min_z_position, self.max_z_position)
    '''
 
    	#if self._is_success(self.state,self.goal)==1 and self.envio==50:
        if self.envio==49:
        	info, position_s = self.sock.RS_send_info(str(action[0]),str(action[1]),str(action[2]),'3')
        	self.envio=0
        else:
        	info, position_s = self.sock.RS_send_info(str(action[0]),str(action[1]),str(action[2]),'1')			#Enviamos al RobotStudio los valores del movimiento en cada eje para que
        	self.envio=self.envio+1                                                                                                                                                                                #lleve a cabo la acción, y nos devuelve el valor de la posición tras dicha acción
        
        positionx, positiony, positionz = string_parser(string = position_s) 



        self.x=positionx
        self.y=positiony
        self.z=positionz



        print(self.envio)
        done=False
        self.state = np.array([self.x,self.y,self.z])
        self.goal=np.array([self.goal_x_position,self.goal_y_position,self.goal_z_position])
        info={'is_success': self._is_success(self.state,self.goal)}
        reward = self.compute_reward(self.state,self.goal,info)
        #return self.state, reward, done, {}
        #print(self.goal)
        #print(positionx,positiony,positionz)
        return {'observation': self.state,'achieved_goal': self.state,'desired_goal':self.goal},reward,done, info

    def reset(self):
        time.sleep(0.1)
        self.envio=1
        info = '0'
        no_se_vale=True

        while info == '0' or no_se_vale:
          self.x = random.uniform(self.min_x_position,self.max_x_position)
          self.y = random.uniform(self.min_y_position, self.max_y_position)
          self.z = random.uniform(self.min_z_position, self.max_z_position)
          self.x = 0.5
          self.y =9.5
          self.z =2
          
          if not funcionlimites(self.x,self.y,self.z):
            info , _ = self.sock.RS_send_info(str(self.x),str(self.y),str(self.z),'0')
            no_se_vale=False
          
          
        info = '0'  
        no_se_vale=True
        while info == '0' or no_se_vale:
          self.goal_x_position = random.uniform(self.min_x_position,self.max_x_position)
          self.goal_y_position = random.uniform(0, 7)
          self.goal_z_position = random.uniform(0, 3.5)
          self.goal_x_position = 9.5
          self.goal_y_position = 2
          self.goal_z_position = 3  
          
          if not funcionlimites(self.goal_x_position,self.goal_y_position,self.goal_z_position):
            info , _ = self.sock.RS_send_info(str(self.goal_x_position),str(self.goal_y_position),str(self.goal_z_position),'2')
            no_se_vale=False
          
        self.state = np.array([self.x,self.y,self.z])
        self.goal=np.array([self.goal_x_position,self.goal_y_position,self.goal_z_position])
        plt.cla()
        self.ax.set_xlim3d(self.min_x_position,self.max_x_position)
        self.ax.set_ylim3d(self.min_y_position,self.max_y_position)
        self.ax.set_zlim3d(self.min_z_position,self.max_z_position)
        self.borrar=0
        return {'observation': self.state.copy(),'achieved_goal': self.state.copy(),'desired_goal': self.goal.copy()}


    def render(self,mode='human'):
        '''self.graficos.delete(ALL)
        self.graficos.create_rectangle(self.goal_x_position*25,self.goal_y_position*25,self.goal_x_position*25+25,self.goal_y_position*25+25,fill='red')
        self.graficos.create_rectangle(self.x*25,self.y*25,self.x*25+25,self.y*25+25,fill='green')
        self.juego.update()
        time.sleep(0.1)'''
        self.ax.scatter3D(self.x, self.y, self.z, c='b')
        self.ax.scatter3D(self.goal_x_position, self.goal_y_position, self.goal_z_position, c='r')
        x0=np.interp(290, [250,550], [0,10])
        y0=np.interp(-150, [-220,220], [0,10])
        z0=np.interp(60, [60,390], [0,10])
        x1=np.interp(527, [250,550], [0,10])
        y1=np.interp(102, [-220,220], [0,10])
        z1=np.interp(240, [60,390], [0,10])

        if self.borrar==0:

            Z = np.array([[x0, y0, z1], [x0, y0, z0], [x0, y1, z0],  [x0, y1, z1], [x1, y0, z1],[x1, y0, z0],[x1, y1, z0],[x1, y1, z1]])
            self.ax.scatter3D(Z[:, 0], Z[:, 1], Z[:, 2])

            # generate list of sides' polygons of our pyramid
            verts = [[Z[0],Z[1],Z[2],Z[3]],
             [Z[4],Z[5],Z[6],Z[7]], 
             [Z[0],Z[1],Z[5],Z[4]], 
             [Z[2],Z[3],Z[7],Z[6]], 
             [Z[1],Z[2],Z[6],Z[5]],
             [Z[4],Z[7],Z[3],Z[0]]]

            # plot sides
            self.ax.add_collection3d(Poly3DCollection(verts, 
             facecolors='cyan', linewidths=1, edgecolors='r', alpha=.10))
            self.borrar=1
        plt.pause(0.1)
        return None 


    def close(self):
        rb.Rs_disconnect()
        return None

