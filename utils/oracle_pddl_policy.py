import numpy as np
import pickle
import copy
from utils.fmm_planner_llm import FMMPlanner
from utils.policy_functions import HeuristicPolicyVer1
from llm.llm import instantiate_llm
import string
import time
import os
import skimage.morphology

class OraclePDDLPolicy(HeuristicPolicyVer1):
	def __init__(self, teleport=False, always_follow_human=False):
		self.hl_tools = ["nav_to_room", "explore_room", "grab_obj", "put_obj", "END_TASK"]
		self.last_return_dict = {}
		self.end_time = 600 
		self.execute_pointer = 0
		self.always_follow_human = always_follow_human

	def decide_whether_execute(self, input_dict):
		task_end_conidition = input_dict['time'] >= self.end_time
		if input_dict['last_tool_ended'] == True or task_end_conidition:
			return True
		else:
			return False 

	def get_cur_pose_room(self, input_dict):
		full_map_pose = input_dict['map_pose_agent']
		cur_pose_room = None
		for r, r_map in input_dict['room_dict']['room_assignment'].items():
			if not(r in ['free_space', 'free space']) and r_map[full_map_pose[0], full_map_pose[1]]==1:
				cur_pose_room = r
		if cur_pose_room is None:  
			full_map = input_dict['full_map']
			planner = FMMPlanner(np.ones(full_map[0].cpu().numpy().shape))
			r_name2dist = {}
			for r, r_map in input_dict['room_dict']['room_assignment'].items():
				if not(r in ['free_space', 'free space']):
					planner.set_multi_goal(r_map)
					dist = planner.fmm_dist[full_map_pose[0], full_map_pose[1]]
					r_name2dist[r] = dist
			#Get the lowest
			room_min = min(r_name2dist, key=lambda k: r_name2dist[k])
			if r_name2dist[room_min] < 0.5 * 100 / 5.:
				cur_pose_room = room_min 
		
		if cur_pose_room == None:
			cur_pose_room = 'free_space'
		return cur_pose_room


	def get_last_action_success(self, input_dict):
		if input_dict['time'] ==0:
			return True
		if self.last_return_dict['hl_tool'] in ["nav_to_room", "explore_room"]:
			success = self.get_cur_pose_room(input_dict) == self.last_return_dict['chosen_room_number']
		#elif self.last_return_dict['hl_tool'] in 
		elif self.last_return_dict['hl_tool']== 'grab_obj':
			success = not(input_dict['failed_interact'])
		elif self.last_return_dict['hl_tool']== 'put_obj':
			success = not(input_dict['failed_interact'])
		elif self.last_return_dict['hl_tool']== 'give_human':
			success = True
		elif self.last_return_dict['hl_tool']== 'follow_human':
			success = input_dict['human_follow_stopped']
		return success


	def execute(self, input_dict):
		if input_dict['time'] ==0:
			self.last_return_dict = {}
			self.execute_pointer = 0

		if input_dict['ignore_cur_ep']:
			self.last_return_dict = {'hl_tool': 'END_TASK', 'whether_execute': True}
			return self.last_return_dict

		if input_dict['task_phase']:
			timeout = (not(self.teleport) and input_dict['time'] >= self.end_time) or (self.teleport and input_dict['execute_count'] >= self.execution_end_time)
			whether_execute = self.decide_whether_execute(input_dict)

			if whether_execute:
				if self.execute_pointer>= len(input_dict['pddl_list']) or timeout:
					self.last_return_dict = {'hl_tool': 'END_TASK', 'whether_execute': True}
					return self.last_return_dict

				last_action_success = self.get_last_action_success(input_dict) 
				if ('hl_tool' in self.last_return_dict) and (self.last_return_dict['hl_tool'] == 'nav_to_room') and not(last_action_success):
					return self.last_return_dict

				#Search if obj not there
				cur_dict = input_dict['pddl_list'][self.execute_pointer]
				if cur_dict['hl_tool'] == 'grab_obj':
					small_objects_inv = {v:k for k,v in input_dict['small_objects'].items()}
					small_obj_numpy = input_dict['full_map'][small_objects_inv[cur_dict['ll_argument']['obj']] + 4].cpu().numpy() 
					selem = skimage.morphology.disk(5)
					so_selem = skimage.morphology.disk(3)
					small_obj_numpy = 1.0 * skimage.morphology.binary_dilation(small_obj_numpy , so_selem)
					room_map = 1.0 * skimage.morphology.binary_dilation(cur_dict['ll_argument']['in_room'] , selem)
					if np.sum(small_obj_numpy  * room_map) ==0:
						self.last_return_dict = {'hl_tool': 'explore_room', 'chosen_room_number': cur_dict['chosen_room_number'], 'll_argument': cur_dict['ll_argument']['in_room']}
						self.last_return_dict['whether_execute'] = whether_execute 
						return self.last_return_dict  

				
				
				if last_action_success:
					self.last_return_dict = input_dict['pddl_list'][self.execute_pointer]
					self.execute_pointer +=1
				else:
					self.last_return_dict = input_dict['pddl_list'][self.execute_pointer]
					self.execute_pointer +=1

			self.last_return_dict['whether_execute'] = whether_execute
			return self.last_return_dict 
		else:
			self.last_return_dict = {}
			return self.last_return_dict 