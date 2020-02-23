import math
import random
import numpy as np
from tkinter import *
import gym
from gym import spaces
from gym.utils import seeding
import time
import matplotlib.pyplot as plt


def goal_distance(goal_a, goal_b):
	return np.linalg.norm(goal_a-goal_b,axis=-1)


class Environment1(gym.Env):
	metadata = {
		'render.modes': ['human', 'rgb_array'],
		'video.frames_per_second': 30
	}

	def __init__(self):
		self.min_x_position = 0
		self.max_x_position = 23
		self.min_y_position = 0
		self.max_y_position = 23
		self.speed = 0.5
		self.goal_x_position = random.randint(self.min_x_position,self.max_x_position)
		self.goal_y_position = random.randint(self.min_y_position,self.max_y_position)
		self.threshold=0.1
		self.x=random.randint(self.min_x_position, self.max_x_position)
		self.y=random.randint(self.min_y_position, self.max_y_position)
		

		self.low = np.array([self.min_x_position, self.min_y_position])
		self.high = np.array([self.max_x_position, self.max_y_position])
		self.low_action = np.array([-self.speed, -self.speed])
		self.high_action = np.array([self.speed, self.speed])
		self.viewer = None

		#self.action_space = spaces.Discrete(4)
		self.action_space = spaces.Box(self.low_action, self.high_action, dtype=np.float32)
		self.observation_space = spaces.Dict(dict(
			desired_goal=spaces.Box(self.low, self.high, dtype='float32'),
			achieved_goal=spaces.Box(self.low, self.high, dtype='float32'),
			observation=spaces.Box(self.low, self.high, dtype='float32')))

		self.seed()
		

		plt.axis([self.min_x_position, self.max_x_position, self.min_y_position, self.max_y_position])



    
		self.juego=Tk()
		self.graficos=Canvas(self.juego,width=600,height=600,bg="pink")
		self.graficos.pack()
		self.graficos.focus_set()
		

	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def compute_reward(self, achieved_goal,goal,info):
		#d=math.sqrt((achieved_goal[0]-goal[0])**2+(achieved_goal[1]-goal[1])**2)
		d=goal_distance(achieved_goal,goal)
		reward = (d>self.threshold)
		return -np.float32(reward)

	def _is_success(self,achieved_goal,desired_goal):
		d=goal_distance(achieved_goal,desired_goal)
		reward = (d<self.threshold)
		return np.float32(reward)

	def step(self, action):
		assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action)) 
		
		positionx, positiony = self.state
		'''if action==0:
			positiony-=self.speed
		if action==2:
			positiony+=self.speed
		if action==1:
			positionx+=self.speed
		if action==3:
			positionx-=self.speed
    	'''
		action[1] = np.clip(action[1], -math.sqrt((self.speed)**2-(action[0])**2), math.sqrt((self.speed)**2-(action[0])**2))
		positionx += action[0]
		positiony += action[1]
		
		positionx = np.clip(positionx, self.min_x_position, self.max_x_position)
		positiony = np.clip(positiony, self.min_y_position, self.max_y_position)
		self.x=positionx
		self.y=positiony

		
		d=math.sqrt((self.x-self.goal_x_position)**2+(self.y-self.goal_y_position)**2)
		#done = bool(d<=self.threshold)
		done=False
		'''
		if done: 
			reward = 1 
		else: 
			reward=-1
		'''


		self.state = np.array([positionx,positiony])
		self.goal=np.array([self.goal_x_position,self.goal_y_position])
		info={'is_success': self._is_success(self.state,self.goal)}
		reward = self.compute_reward(self.state,self.goal,info)
		#return self.state, reward, done, {}
		return {'observation': self.state,'achieved_goal': self.state,'desired_goal':self.goal},reward,done, info

	def reset(self):
		self.x = random.randint(self.min_x_position,self.max_x_position)
		self.y = random.randint(self.min_y_position, self.max_y_position)
		self.goal_x_position = random.randint(self.min_x_position,self.max_x_position)
		self.goal_y_position = random.randint(self.min_y_position,self.max_y_position)
		self.state = np.array([self.x,self.y])
		self.goal=np.array([self.goal_x_position,self.goal_y_position])
		plt.clf()
		plt.axis([self.min_x_position, self.max_x_position, self.min_y_position, self.max_x_position])
		return {'observation': self.state.copy(),'achieved_goal': self.state.copy(),'desired_goal': self.goal.copy()}


	def render(self,mode='human'):
		'''self.graficos.delete(ALL)
		self.graficos.create_rectangle(self.goal_x_position*25,self.goal_y_position*25,self.goal_x_position*25+25,self.goal_y_position*25+25,fill='red')
		self.graficos.create_rectangle(self.x*25,self.y*25,self.x*25+25,self.y*25+25,fill='green')
		self.juego.update()
		time.sleep(0.1)'''
		plt.scatter(self.goal_x_position, self.goal_y_position)
		plt.scatter(self.x, self.y)
		plt.pause(0.05)
		return None 

	def close(self):
		return None