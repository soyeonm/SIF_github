import numpy as np
import pickle
import copy
from utils.fmm_planner_llm import FMMPlanner
from utils.policy_functions import HeuristicPolicyVer1
from utils.get_human_walk_room_function import get_human_walk_towards_rooms
from llm.llm import instantiate_llm
import string
import time
import os
import skimage.morphology
import random


class PrompterPolicy(HeuristicPolicyVer1):
	def __init__(self, llm_type = 'openai', host=None, baseline_type='reasoning', teleport=False, magic_grasp=False, add_communication=True, human_trajectory_index=None, add_human_loc=False):
		#avaialble high level tools
		self.hl_tools = ["nav_to_room", "explore_room", "grab_obj", "put_obj", "END_TASK"]
		self.last_return_dict = {}
		self.end_time = 600 
		self.teleport = teleport
		self.llm_type = llm_type
		if llm_type=='openai':
			self.prompter =  instantiate_llm('openai_chat', generation_params={'model':'gpt-3.5-turbo-0125'})
		elif llm_type == 'openai_chat':
			raise Exception("Not Implemented")
		else:
			self.prompter =  instantiate_llm(llm_type, host='learnfair'+host)
		self.environment_prompts = {} 
		self.llm_type = llm_type
		self.llama_in_contxt_example = 'Example Task:\n You are an assistive robot in a house, and a human in the house has walked around in the house, moving some objects. Your goal is to find and rearrange objects as in the instruction “Find the Candle”.\n Here are your observations across time. Timestep 0: You did frontier-based exploration and did an initial scan of the house, before the human moved objects. You made a map of the house in this exploration - we will call this map your “Map 0” (map made at tilmestep 0). \n After timestep 0,.You were standing in the living room (Room 0) and you have partially observed the human moving.\n  Timestep 1: The human moved inside Room 0, holding a candle. The human put the candle from the chair to the countertop.\n Your goal is to find the candle. You should only respond in the format as described below: RESPONSE FORMAT: Reasoning: Based on the information I listed above, do reasoning about what next room to search. Room: Next Room.\n Now, which room should you go to? Please answer with the RESPONSE FORMAT.\n RESPONSE FORMAT:\n Reasoning: Based on the information I listed above, I observed the human move the candle inside Room 0 (the living room) and place it from the chair to the countertop. Therefore, I do not need to search another room since I already know the location of the candle in Room 0.\n Room: Room 0.\n Actual Task: '
		self.baseline_type = baseline_type
		self.magic_grasp = magic_grasp
		self.invalid_receps = set(['bench', 'fridge', 'shelves', 'unidentified_recep'])
		self.call_limit = 15 
		self.add_communication = True
		self.human_trajectory_index = human_trajectory_index
		self.add_human_loc = add_human_loc

	def get_current_map_prompt(self, input_dict,map_lo_so_dict_input,  state=None, current_room_only=None):
		map_prompt = map_to_prompt_recep_only(map_lo_so_dict_input, input_dict, state, current_room_only)
		return map_prompt

	def generate_action_set(self, input_dict):
		map_lo_so_dict = self.most_recent_map_lo_so_dict 
		objs = []
		for k, v in map_lo_so_dict.items():
			if not(k == "free_space"):
				for recep, value_list in map_lo_so_dict[k].items():
					objs += value_list
		
		manipulation_objs =  []
		if not(input_dict['is_holding'] == None):
			for k, v in map_lo_so_dict.items():
				if not(k == "free_space"):
					manipulation_objs += ["Put " + "on the " +recep for recep in map_lo_so_dict[k] if (recep!='unidentified_recep' and recep!='human' and not(recep in self.invalid_receps))]
		else:
			manipulation_objs += ["Pick up " + obj for obj in objs]
		return manipulation_objs

	def generate_representation(self, input_dict, task_type):
		rooms = []
		for r, r_map in input_dict['room_dict']['room_assignment'].items():
			if r!='free_space' and r!='free space':
				rooms.append(r)

		self.system_prompt = get_system_prompt(input_dict)
		self.fbe_map_prompt = "While premapping, you saw the layout of the house; " + self.get_current_map_prompt(input_dict, input_dict['fbe_map_lo_so_dict'], state='premap').replace('While premapping,', '')
		self.goal_prompt = " The human commanded to you: '" + input_dict['prompt']['initial']  + ".'" 
		self.goal_only_goal_prompt = self.goal_prompt 
		self.evidence_prompt =  " Right after the human said this, you saw the human in room " +  str(input_dict['human_pose_room_initial']) + ". Now, from the human's utterance (" + str(input_dict['prompt']['initial']) + ") and your observations of the human, do you have enough evidence which room (among rooms" + ", ".join([str(r) for r in rooms]) + " ) the human is walking towards and will stop in?"
		self.evidence_prompt += " Please choose among ['Enough Evidence', 'Not Enough Evidence']."
		if task_type == 'recep':
			self.goal_prompt+= 'Please make a high level plan. Please answer in the format of "[Pick up OBJ, Put on the RECEP]". For example, "[Pick up saltshaker, Put on the bed]".'
		elif task_type == 'room':
			self.goal_prompt+= 'Please make a high level plan. Please answer in the format of "[Pick up OBJ, Put in the ROOM]". For example, "[Pick up saltshaker, Put in the bedroom]".'
		elif task_type == 'human':
			self.goal_prompt+= 'Please make a high level plan. Please answer in the format of "[Pick up OBJ, Give Human]". For example, "[Pick up saltshaker, Give Human]".'
		else:
			raise Exception("Invliad tasktype!")

		return self.system_prompt, self.fbe_map_prompt, self.goal_prompt, self.evidence_prompt

	###########################
	def are_strings_equal(self, str1, str2):
	    normalized_str1 = str1.replace(' ', '').replace('_', '')
	    normalized_str2 = str2.replace(' ', '').replace('_', '')
	    return normalized_str1 == normalized_str2

	def reconstruct_action_output(self, input_dict, action_output):
		action_output = ''.join(c for c in action_output if c not in string.punctuation.replace('_', '').replace('!', ''))
		try:
			if 'Pick up ' in action_output:
				#Then return pickup tool
				obj_entity = action_output.replace('Pick up ', '')
				reconstruct_action_output = 'Pick up ' + obj_entity
			elif "Put on " in action_output:
				recep = action_output.split("on the ")[-1].replace(".", "").replace("]", "")
				reconstruct_action_output = "Put on the " +recep 
			elif "Put in " in action_output:
				room = action_output.split("in the ")[-1].replace(".", "").replace("]", "")
				reconstruct_action_output = "Put in the " +room 
			else:
				reconstruct_action_output = action_output
		except:
			reconstruct_action_output = action_output

		
		for ca in self.cur_action_set:
			if self.are_strings_equal(ca, reconstruct_action_output):
				return ca
		return reconstruct_action_output

	def parse_room_result(self, room_string):
		if 'Room ' in room_string:
			try:
				room_num= int(room_string.replace('.', '').split('Room ')[1].split(',')[0])
			except:
				room_num = None
		else:
			room_num = None
		return room_num

	def get_cur_pose_room(self, input_dict):
		full_map_pose = input_dict['map_pose_agent']
		cur_pose_room = None
		for r, r_map in input_dict['room_dict']['room_assignment'].items():
			if r!='free_space' and r!='free space' and r_map[full_map_pose[0], full_map_pose[1]]==1:
				cur_pose_room = r
		if cur_pose_room is None:
			full_map = input_dict['full_map']
			planner = FMMPlanner(np.ones(full_map[0].cpu().numpy().shape)) 
			r_name2dist = {}
			for r, r_map in input_dict['room_dict']['room_assignment'].items():
				if r!='free_space' and r!='free space':
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

	###########################
	def execute(self, input_dict):
		#Reset environment prompts
		if self.llm_type=='openai':
			self.prompter =  instantiate_llm('openai_chat', generation_params={'model':'gpt-3.5-turbo-0125'})
		elif self.llm_type == 'openai_chat':
			raise Exception("Does not support!")
		
		if not(input_dict['task_phase']):
			self.last_return_dict = {}
			return self.last_return_dict 

		if input_dict['ignore_cur_ep']:
			return {'hl_tool': 'END_TASK', 'whether_execute': True}

		if input_dict['time'] >= self.end_time:
			return {'hl_tool': 'END_TASK', 'whether_execute': True}


		if input_dict['time']==0:
			self.follow_not_ended = True
			self.follow_true = False
			self.environment_prompts = {}
			self.grabbed_obj = None
			self.enough_evidence = False
			self.last_return_dict = {}
			self.most_recent_map_lo_so_dict = get_map_lo_so_dict(input_dict, include_human=True)
			self.cur_action_set = self.generate_action_set(input_dict) 
			#Get tasktype
			self.system_prompt, self.fbe_map_prompt, self.goal_prompt, self.evidence_prompt = self.generate_representation(input_dict, input_dict['task_type'])
			

			if input_dict['task_type'] != 'human':
				self.high_level_plan_string = self.prompter.generate(self.goal_prompt)
				self.high_level_plan = self.high_level_plan_string.split(',')
				
			else:
				self.evidence_plan_string = self.prompter.generate(self.system_prompt + ' ' + self.fbe_map_prompt + self.goal_only_goal_prompt + ' ' + self.evidence_prompt)
				self.high_level_plan_string = self.prompter.generate(self.goal_prompt)
				self.high_level_plan = self.high_level_plan_string.split(',')
				if 'Not' in self.evidence_plan_string:
					self.follow_true = True
					self.high_level_plan = ["Follow Human"] + self.high_level_plan
				else:
					pass
			
			if self.follow_true:
				self.pickup_action_output = self.reconstruct_action_output(input_dict,self.high_level_plan[1])
			else:
				self.pickup_action_output = self.reconstruct_action_output(input_dict,self.high_level_plan[0])
			self.put_give_action_output = self.reconstruct_action_output(input_dict,self.high_level_plan[-1])
			self.done = False
			self.last_room_search = None
			self.num_called = 0
			self.max_human_index = -1
			self.give_human_declared = False
			self.last_room_number = None

		if input_dict['time']>0:
			self.cur_action_set = self.generate_action_set(input_dict) 
			if self.last_return_dict['hl_tool'] == "nav_directly_to_goal_put" and input_dict['last_tool_ended']:
				room_number = "Done declared!"
				return {'hl_tool': 'END_TASK', 'whether_execute': True, 'chosen_room_number':None, 'llm_rep': None, 'gen_string': None}
			if self.last_return_dict['hl_tool'] == 'follow_human' and input_dict['last_tool_ended']:
				self.follow_not_ended = False

		if self.follow_true and self.follow_not_ended and input_dict['last_tool_ended']:
			hl_tool = 'follow_human'
			room_number = 0
			ask_again_every_20 = False
			ll_arg = {'ask_again_every_20': ask_again_every_20}
			self.last_return_dict= {'hl_tool': hl_tool, 'll_argument': ll_arg, 'whether_execute': True,
													'llm_rep': (self.system_prompt + self.fbe_map_prompt + self.goal_only_goal_prompt + self.evidence_prompt, self.goal_prompt),
													'gen_string': (self.evidence_plan_string, self.high_level_plan_string),
													'action_output': None,
													'reasoning_output': None,
													'chosen_room_number': None}
			return self.last_return_dict

		else:
			#Do pickup
			if not(input_dict['is_holding']):
				if self.pickup_action_output  in self.cur_action_set:
					pickup_obj = self.pickup_action_output.replace("Pick up ", "")
					small_objects_inv = {v:k for k,v in input_dict['small_objects'].items()}
					small_obj_numpy = input_dict['full_map'][small_objects_inv[pickup_obj] + 4].cpu().numpy() 
					
					if np.sum(small_obj_numpy) ==0:
						if self.prev_call_search and self.last_return_dict['hl_tool'] == 'nav_to_room' and input_dict['last_tool_ended']:
							hl_tool = "explore_room"
							self.last_return_dict['whether_execute'] = True
							self.last_return_dict['hl_tool'] = "explore_room"
						elif  input_dict['last_tool_ended']:
							if self.num_called >= self.call_limit:
								room_number = "called more than 30 times!"
								return {'hl_tool': 'END_TASK', 'chosen_room_number':room_number, 'whether_execute': True, 'gen_string': self.high_level_plan_string, 'llm_rep': (self.goal_prompt, self.ask_prompt)}
		
							
							question_prompt = "\n In which room, is the " + pickup_obj + " likely to be in? Please answer in the format of Room X. For example, Room 1."
							question_prompt  +="Please ONLY respond in the format: Answer: Room X."
							self.prompter =  instantiate_llm('openai_chat', generation_params={'model':'gpt-3.5-turbo-0125'}, temperature=0.3, top_p=0.3)
							interm_prompt =  ""

							if not(self.add_communication):
								room_string = self.prompter.generate(self.system_prompt + self.fbe_map_prompt + question_prompt)
							else:
								interm_prompt = ""
								if len(input_dict['prompt']['intermediate'])> 0:
									interm_prompt = "The human had said "
									interm_prompt += "; ".join(["'"+ p['prompt']['sentences'] + "'." for p in input_dict['prompt']['intermediate']])
								room_string = self.prompter.generate(self.system_prompt + self.fbe_map_prompt + interm_prompt + question_prompt )
							self.ask_prompt = self.system_prompt + self.fbe_map_prompt + interm_prompt + question_prompt 
							self.num_called +=1
							room_num = self.parse_room_result(room_string)
							room_number =  room_num
							if self.last_room_search == None:
								hl_tool = "nav_to_room"
							elif self.last_room_search == room_num:
								hl_tool = "explore_room"
							else:
								hl_tool = "nav_to_room"

							
							self.last_return_dict['whether_execute'] = True
							reasoning_output = ""
							if room_num == None:
								room_number = "couldn't parse action output"
								return {'hl_tool': 'END_TASK', 'chosen_room_number':room_number, 'whether_execute': True, 'gen_string': self.high_level_plan_string + '\n ' + room_string, 'llm_rep': (self.goal_prompt, self.ask_prompt)}

							else:
								ll_arg = room_goal(input_dict['time'], input_dict['room_dict']['room_assignment'], room_number)
								self.last_return_dict = {'hl_tool': hl_tool, 'll_argument': ll_arg, 'whether_execute': True,
												'llm_rep': (self.goal_prompt, self.ask_prompt),
												'gen_string': self.high_level_plan_string + '\n ' + room_string,
												'action_output': room_number,
												'reasoning_output': reasoning_output,
												'chosen_room_number': room_number}

						else:
							self.last_return_dict['whether_execute'] = False

						
						self.prev_call_search = True
						self.last_room_search = self.last_return_dict['action_output']
						return self.last_return_dict
					else:
						self.prev_call_search = False
						hl_tool = "nav_directly_to_goal_pick"
						ll_arg = {'obj': pickup_obj}
						if input_dict['last_tool_ended']:
							self.last_return_dict= {'hl_tool': hl_tool, 'll_argument': ll_arg, 'whether_execute': True,
													'llm_rep': (self.goal_prompt),
													'gen_string': self.high_level_plan_string,
													'action_output': None,
													'reasoning_output': None,
													'chosen_room_number': None}
							return self.last_return_dict
						else:
							self.last_return_dict['whether_execute'] = False
							#breakpoint()
							return self.last_return_dict

				else:
					room_number = "couldn't parse action output"
					return {'hl_tool': 'END_TASK', 'chosen_room_number':room_number, 'whether_execute': True, 'gen_string': self.high_level_plan_string, 'llm_rep':(self.goal_prompt)}
				
			else:
				self.prev_call_search = False
				if (self.put_give_action_output in self.cur_action_set) or ("Put in " in self.put_give_action_output):
					if input_dict['task_type'] in ['room', 'recep']:		
						if input_dict['last_tool_ended']:
							r = None
							if input_dict['task_type'] == 'room':
								recep = None
								room = self.put_give_action_output.replace("Put in the ", "")
								if self.num_called >= self.call_limit:
									room_number = "called more than 30 times!"
									return {'hl_tool': 'END_TASK', 'chosen_room_number':room_number, 'whether_execute': True, 'gen_string': self.high_level_plan_string + '\n ', 'llm_rep': (self.goal_prompt)}

								room_ground_question_prompt = "Which room is the most likely to be the " + room + "? Please answer in the format of Room X. For example, Room 1."
								room_ground_question_prompt  +=" Please ONLY respond in the format: Answer: Room X."
								self.prompter =  instantiate_llm('openai_chat', generation_params={'model':'gpt-3.5-turbo-0125'}, temperature=0.3, top_p=0.3)
								room_gen_string = self.prompter.generate(self.system_prompt + self.fbe_map_prompt + room_ground_question_prompt)
								self.num_called +=1
								r = int(room_gen_string.replace(".", "").split('Room ')[-1])
								if not(r in input_dict['fbe_map_lo_so_dict']):
									room_number = "couldn't parse action output"
									return {'hl_tool': 'END_TASK', 'chosen_room_number':room_number, 'whether_execute': True, 'gen_string': self.high_level_plan_string, 'llm_rep':(self.goal_prompt)}

								#Just sample a recep inside the room
								invalid_receps = set(['bench', 'fridge', 'shelves', 'unidentified_recep'])
								goal_recep_room = None
								val_k = [k  for k in input_dict['fbe_map_lo_so_dict'][r] if not(k in invalid_receps)]
								if len(input_dict['fbe_map_lo_so_dict'][r])> 0 and len(val_k)>0:
									goal_recep_room = r
									random.seed(input_dict['time']+10)
									recep = random.choice(val_k)

								if recep == None:
									room_number = "couldn't parse action output"
									return {'hl_tool': 'END_TASK', 'chosen_room_number':room_number, 'whether_execute': True, 'gen_string': self.high_level_plan_string, 'llm_rep':(self.goal_prompt)}

							elif input_dict['task_type'] == 'recep':
								recep = self.put_give_action_output.replace("Put on the ", "")


							hl_tool = "nav_directly_to_goal_put"
							ll_arg = {'obj': recep}
							self.last_return_dict= {'hl_tool': hl_tool, 'll_argument': ll_arg, 'whether_execute': True,
													'llm_rep': (self.goal_prompt),
													'gen_string': self.high_level_plan_string,
													'action_output': None,
													'reasoning_output': None,
													'chosen_room_number': r}
							return self.last_return_dict
						else:
							self.last_return_dict['whether_execute'] = False
							#breakpoint()
							return self.last_return_dict

				else: #give human
					human_recep = input_dict['prompt']['initial_recep_type_human']
					if ("Give Human" in self.put_give_action_output ) and human_recep:
						if input_dict['last_tool_ended']:
							cur_pose_room = self.get_cur_pose_room(input_dict)

							if self.give_human_declared :
								room_number = "Done!"
								return {'hl_tool': 'END_TASK', 'chosen_room_number':cur_pose_room, 'whether_execute': True, 'gen_string': self.high_level_plan_string, 'llm_rep':(self.goal_prompt)}
							
							if human_recep and not(cur_pose_room == "free_space") and (cur_pose_room == input_dict['human_walking_to_room']) and not(input_dict['is_holding'] == None): #and not(self.grabbed_obj == None):
								self.give_human_declared = True
								return {'hl_tool': 'give_human', 'chosen_room_number':cur_pose_room, 'whether_execute': True, 'gen_string': self.high_level_plan_string, 'llm_rep':(self.goal_prompt)}

							
							where_human = input_dict['full_map'][4+self.human_trajectory_index].detach().cpu().numpy()
							max_human_index = np.max(where_human )
							room_number = input_dict['human_pose_room_cur'] 
							human_ask_prompt = ''

							if max_human_index > self.max_human_index:
								ll_arg = 1.0 * (where_human == max_human_index)
								self.max_human_index = max_human_index
								self.last_room_number = room_number
								self.last_return_dict = {'hl_tool': 'nav_to_room', 'll_argument': ll_arg, 'whether_execute': True,
												'llm_rep': (self.goal_prompt, human_ask_prompt),
												'gen_string': self.high_level_plan_string,
												'action_output': room_number,
												'reasoning_output': None,
												'chosen_room_number': room_number}

							elif max_human_index == self.max_human_index:
								room_ground_question_prompt = "\n Which room is the human most likely to be in? Please answer in the format of Room X. For example, Room 1."
								self.prompter =  instantiate_llm('openai_chat', generation_params={'model':'gpt-3.5-turbo-0125'}, temperature=0.3, top_p=0.3)
								
								if self.add_human_loc:
									human_ask_prompt = self.system_prompt + self.goal_only_goal_prompt + ' ' + self.fbe_map_prompt + room_ground_question_prompt
									room_string = self.prompter.generate(self.system_prompt + self.goal_only_goal_prompt + ' ' + self.fbe_map_prompt + room_ground_question_prompt)
								else:
									human_ask_prompt = self.system_prompt + self.fbe_map_prompt + room_ground_question_prompt
									room_string = self.prompter.generate(self.system_prompt + self.fbe_map_prompt + room_ground_question_prompt)
								self.num_called +=1
								room_num = self.parse_room_result(room_string)
								room_number = room_num
								self.last_room_number = room_number
								self.max_human_index = max_human_index
								if self.num_called >= self.call_limit:
									room_number = "called more than 30 times!"
									return {'hl_tool': 'END_TASK', 'chosen_room_number':room_num, 'whether_execute': True, 'gen_string': self.high_level_plan_string + '\n ' + room_string, 'llm_rep': (self.goal_prompt)}

								if room_num == None:
									room_number = "couldn't parse action output"
									return {'hl_tool': 'END_TASK', 'chosen_room_number':room_number, 'whether_execute': True, 'gen_string': self.high_level_plan_string + '\n ' + room_string, 'llm_rep': (self.goal_prompt, self.ask_prompt)}


								ll_arg = room_goal(input_dict['time'], input_dict['room_dict']['room_assignment'], room_number)
								self.last_return_dict = {'hl_tool': 'nav_to_room', 'll_argument': ll_arg, 'whether_execute': True,
												'llm_rep': (self.goal_prompt,  human_ask_prompt),
												'gen_string': self.high_level_plan_string + '\n ' + room_string,
												'action_output': room_number,
												'reasoning_output': None,
												'chosen_room_number': room_number}
							return self.last_return_dict

						else:
							self.last_return_dict['whether_execute'] = False
							return self.last_return_dict

					else:
						room_number = "couldn't parse action output"
						return {'hl_tool': 'END_TASK', 'chosen_room_number':room_number, 'whether_execute': True, 'gen_string': self.high_level_plan_string, 'llm_rep':(self.goal_prompt)}







		

		

def large_objects_into_clusters(input_dict):
	room_dict = input_dict['room_dict']['room_assignment']
	large_objects = input_dict['large_objects'] 
	full_map = input_dict['full_map'].detach().cpu().numpy()
	large_objects_connected_regions = {}
	for l in large_objects:
		if l>0 : 
			connected_regions = skimage.morphology.label(full_map[l+4], connectivity=2)
			for c in np.unique(connected_regions):
				if c>0: 
					for room_num, room_map in room_dict.items():
						if np.sum(room_map * (connected_regions==c)):
							if not(l in large_objects_connected_regions):
								large_objects_connected_regions[l] = {}
							if not(room_num in large_objects_connected_regions[l]):
								large_objects_connected_regions[l][room_num] = np.zeros(room_map.shape) #[]
							large_objects_connected_regions[l][room_num] += (connected_regions==c)
							large_objects_connected_regions[l][room_num] = 1.0 *(large_objects_connected_regions[l][room_num]>0)
	return large_objects_connected_regions


def map_to_prompt_recep_only(map_lo_so_dict, input_dict, state=None, current_room_only=None):
	room_2_objects = map_lo_so_dict
	room_count = 0
	header = ""
	for r in room_2_objects:
		room_count +=1
		if r!='free_space' and r!='free space':
			if len(room_2_objects[r]) >0:
				header += " Room " + str(r) + " has: "
				lo_count = 0
				for lo in room_2_objects[r]:
					if not(lo == 'unidentified_recep'):
						header += lo + ", "
				if room_count < len(room_2_objects[r]):
					header +=";"
				else:
					header += "."
	return header

#Representation functions
def map_to_prompt(map_lo_so_dict, input_dict, state=None, current_room_only=None):
	room_2_objects = map_lo_so_dict
	time_step = input_dict['time']
	if state == 'premap':
		header = "While premapping, you saw that"
	elif state == 'scripted':
		if input_dict['baseline_type'] == 'reasoning':
			header = "After this, based on your latest observations, in your map, "
		elif input_dict['baseline_type'] == 'film':
			header = "Based on your latest observations, in your map, "
	else:
		header = "In map " + str(time_step) + " (at time step " + str(time_step) + "), you saw that"
	if current_room_only != None:
		header =  "Based on the preamap and the parts of the rooms that you scanned just now, in map "+ str(time_step) + " (at time step " + str(time_step) + "), you saw that"
	room_count = 0
	if current_room_only == None:
		for r in room_2_objects:
			room_count +=1
			if r!='free_space' and r!='free space':
				if len(room_2_objects[r]) >0:
					header += " Room " + str(r) + " has: "
					lo_count = 0
					
					for lo in room_2_objects[r]:
						if not(lo == 'unidentified_recep'):
							header += lo + ", "
							for so in room_2_objects[r][lo]:
								header += so + " on a " + lo + ", " 
							lo_count +=1
						else:
							for so in room_2_objects[r][lo]:
								header += so + " on some receptacle " + ", " 
					if room_count < len(room_2_objects[r]):
						header +=";"
					else:
						header += "."
	else:
		r = current_room_only
		if r!='free_space' and r!='free space':
			if len(room_2_objects[r]) >0:
				header += " Room " + str(r) + " has: "
				lo_count = 0
				
				for lo in room_2_objects[r]:
					if not(lo == 'unidentified_recep'):
						header += lo + ", "
						for so in room_2_objects[r][lo]:
							header += so + " on a " + lo + ", " 
						lo_count +=1
					else:
						for so in room_2_objects[r][lo]:
							header += so + " on some receptacle " + ", " 
				if room_count < len(room_2_objects[r]):
					header +=";"
				else:
					header += "."
		else:
			header = " Room " + str(r) + " has nothing."
	header += " Your observations (and the map) can be stale/ incorrect/ missing, because the human moved objects since you had premapped the house, or your vision system is imperfect."
	header = header.replace(', ;', ';')
	header = header.replace(', .', '.')
	return header


def get_system_prompt(input_dict): 
	prompt = 'You are an assistive robot in a house, helping a human.' 
	prompt += ' Your observations may be incomplete or wrong.' 
	return prompt

def get_format_prompt_human_follow(input_dict, cur_action_set, map_lo_so_dict, cur_pose_room):
	objects_in_room = []
	if cur_pose_room !='free_space':
		for v in map_lo_so_dict[cur_pose_room].values():
			objects_in_room +=v 
	human_recep = input_dict['prompt']['initial_recep_type_human']

	prompt = 'The human said your goal is to “' + input_dict['prompt']['initial'] + '”' + '. ' + 'The human may be nuanced or ambiguous.' 
	if human_recep:
		human_pose_room_cur = input_dict['human_pose_room_cur']
		human_pose_room_initial = input_dict['human_pose_room_initial']
		human_towards_rooms = input_dict['human_towards_rooms']
		prompt += " Right after the human said this, you saw the human in room " +  str(human_pose_room_initial) + ", starting to walk towards somewhere. When you last saw the human, the human was in room " + str(human_pose_room_cur)
		if input_dict['human_follow_stopped']:
			prompt += ". You saw that the human has stopped here or either you lost track of the human.\n"
		else:
			if not(human_towards_rooms is None) and len(human_towards_rooms) >0:
				prompt += ", walking towards rooms " + ", ".join([str(h) for h in human_towards_rooms]) + ".\n"
			else:
				prompt += ". \n"
	rooms = []
	for r, r_map in input_dict['room_dict']['room_assignment'].items():
		if r!='free_space' and r!='free space':
			rooms.append(r)

	human_obs_prompt = "When you last saw the human, the human was in room " + str(human_pose_room_cur)
	if input_dict['human_follow_stopped']:
		human_obs_prompt += ". You saw that the human has stopped here or either you lost track of the human."
	else:
		if not(human_towards_rooms is None) and len(human_towards_rooms) >0:
			human_obs_prompt += ", walking towards rooms " + ", ".join([str(h) for h in human_towards_rooms]) + "."
		else:
			pass


	prompt += "Now, from the human's utterance (" + str(input_dict['prompt']['initial']) + ") and your observations of the human (" + human_obs_prompt + "), do you have enough evidence which room (among rooms" + ", ".join([str(r) for r in rooms]) + " ) the human is walking towards and will stop in?"
	prompt +="Please ONLY respond in the format: RESPONSE FORMAT: Reasoning: reason about 'Enough Evidence'/ 'Not Enough Evidence'. Action: choose among ['Enough Evidence', 'Not Enough Evidence']."
	return prompt

def get_format_prompt(input_dict, cur_action_set, map_lo_so_dict, cur_pose_room):
	objects_in_room = []
	if cur_pose_room !='free_space':
		for v in map_lo_so_dict[cur_pose_room].values():
			objects_in_room +=v 
	human_recep = input_dict['prompt']['initial_recep_type_human']

	prompt = 'The human said your goal is to “' + input_dict['prompt']['initial'] + '”' + '. ' + 'The human may be nuanced or ambiguous.' 
	if human_recep:
		human_pose_room_cur = input_dict['human_pose_room_cur']
		human_pose_room_initial = input_dict['human_pose_room_initial']
		human_towards_rooms = input_dict['human_towards_rooms']
		prompt += " Right after the human said this, you saw the human in room " +  str(human_pose_room_initial) + ", starting to walk towards somewhere. When you last saw the human, the human was in room " + str(human_pose_room_cur)
		if input_dict['human_follow_stopped']:
			prompt += ". You saw that the human has stopped here or either you lost track of the human.\n"
		else:
			if not(human_towards_rooms is None) and len(human_towards_rooms) >0:
				prompt += ", walking towards rooms " + ", ".join([str(h) for h in human_towards_rooms]) + ".\n"
			else:
				prompt += ". \n"
	prompt += " To achieve the goal, choose the next action among Available Actions: "+ str(cur_action_set) + "' .\n"
	prompt += "You are in room " + str(cur_pose_room) + " and you can ONLY grab/ put objects inside Room " + str(cur_pose_room) #". These objects are: " + str(list(map_lo_so_dict[cur_pose_room].keys()) + objects_in_room)  + ", since they are in room " + str(cur_pose_room) + " . You CANNOT put to/ grab other objects.\n"
	if str(cur_pose_room != 'free_space'):
		if input_dict['baseline_type'] == 'film':
			prompt += ". Use your common sense of where objects are likely to be and search for objects of interest "
		prompt += ". If you want to keep searching for object(s) or human that might exist (but you have not detected) in the current room, choose 'Explore Room " + str(cur_pose_room) + "'. To go to another room, choose 'Go to Room '" + "another room number'." 
		prompt += "\n "
	prompt +="Please ONLY respond in the format: RESPONSE FORMAT: Reasoning: reason about the next action. Action: one of Available Actions (e.g." +str(cur_action_set[0]) + "). If task is complete, answer Action: Done!\n Now, which action should you take? RESPONSE FORMAT:\n"
	return prompt

def second_chance_prompt(input_dict, cur_action_set):
	prompt = "Please ONLY choose among the Current Available Action Set "+ str(cur_action_set) + "' .\n"
	prompt +="ONLY respond in this format: RESPONSE FORMAT: Reasoning: reason about what next action to take. Action: One of the Current Available Action Set (e.g." +str(cur_action_set[0]) + "). If you complete the task, answer Action: Done!\n Now, which action should you take? RESPONSE FORMAT:\n"
	return prompt


def get_map_lo_so_dict(input_dict, include_human):
	room_dict = input_dict['room_dict']['room_assignment']
	full_map = input_dict['full_map'].detach().cpu().numpy() 

	small_objects = input_dict['small_objects']	
	large_objects = input_dict['large_objects']	
	human_idx = input_dict['human_index']
	room_2_objects = {r: {} for r in room_dict}
	selem = skimage.morphology.disk(5)
	so_selem = skimage.morphology.disk(3)
	for room_num, room_map_ori in room_dict.items():
		room_map = copy.deepcopy(room_map_ori)
		room_map = skimage.morphology.binary_dilation(room_map, selem)
		for lo, lo_cat in large_objects.items():
			if lo>0:
				lo_map = full_map[lo+4]
				if np.sum(lo_map * room_map) >0:
					room_2_objects[room_num][lo_cat] = []
					for so, so_cat in small_objects.items():
						so_map = copy.deepcopy(full_map[so+4])
						so_map = skimage.morphology.binary_dilation(so_map, selem)
						if np.sum(lo_map * room_map * so_map) >0:
							room_2_objects[room_num][lo_cat].append(so_cat)

	for room_num, room_map in room_dict.items():
		for so, so_cat in small_objects.items():
			so_map = full_map[so+4]
			if np.sum(room_map * so_map) >0:
				s_list = []
				for v in room_2_objects[room_num].values():
					s_list += v
				if not(so_cat in s_list):
					if not ('unidentified_recep' in room_2_objects[room_num]):
						room_2_objects[room_num]['unidentified_recep'] = []
					room_2_objects[room_num]['unidentified_recep'].append(so_cat)


	if include_human:
		for room_num, room_map in room_dict.items():
			if np.sum(full_map[human_idx+4] * room_map)>0:
				room_2_objects[room_num]['human'] = []


	for k, v in room_2_objects.items():
		if not(k in ['free_space', 'free space']):
			rf = input_dict['room_dict']['human_annotation'][k]['function']
			if rf!='bathroom' and ('toilet' in v):
				v.pop('toilet')
	if 'free_space' in room_2_objects:
		room_2_objects.pop('free_space')
	if 'free space' in room_2_objects:
		room_2_objects.pop('free space')
	return room_2_objects

def room_goal(seed, room_dict, room, explore_room=False):
	goal= room_dict[room] * 1.0 
	if not(explore_room):
		selem = skimage.morphology.disk(20) #1.0 meter
		goal = skimage.morphology.binary_erosion(goal, selem)
		if np.sum(goal) ==0:
			goal_dilated= room_dict[room] * 1.0
			selem = skimage.morphology.disk(1)
			while np.sum(goal_dilated) >10:
				goal = copy.deepcopy(goal_dilated)
				goal_dilated = skimage.morphology.binary_erosion(goal, selem)
	return goal*1.0

def select_random_coord_in_room(seed, room_dict, room):
	#randomly choose a room 
	np.random.seed(seed)
	#randomly chooose a goal inside the room
	wheres = np.where(room_dict[room])
	len_where = len(wheres[0])
	np.random.seed(seed)
	where_idxes = np.random.choice(np.arange(len_where), 10)
	global_goals = [[wheres[0][where_idx], wheres[1][where_idx]] for where_idx in where_idxes]
	return global_goals