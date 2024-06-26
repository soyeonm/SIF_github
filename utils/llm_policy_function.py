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


#Only get nav to room for now
class Reasoner(HeuristicPolicyVer1):
	def __init__(self, llm_type = 'openai', host=None, baseline_type='reasoning', teleport=False, magic_grasp=False, human_recep_no_put=False):
		#avaialble high level tools
		self.hl_tools = ["nav_to_room", "explore_room", "grab_obj", "put_obj", "END_TASK"]
		self.last_return_dict = {}
		self.end_time = 600 #300
		self.teleport = teleport
		self.execution_end_time = 15 #3 #15
		self.llm_type = llm_type
		if llm_type == 'replay':
			raise Exception("LLM Type replay")
		elif llm_type=='openai':
			self.reasoner =  instantiate_llm('openai_chat', generation_params={'model':'gpt-3.5-turbo-0125'})
		elif llm_type == 'openai_chat':
			self.reasoner =  instantiate_llm(llm_type, generation_params={'model':'gpt-4o'}) 
		else:
			self.reasoner =  instantiate_llm(llm_type, host='learnfair'+host)
		self.environment_prompts = {} 
		self.llm_type = llm_type
		self.llama_in_contxt_example = 'Example Task:\n You are an assistive robot in a house, and a human in the house has walked around in the house, moving some objects. Your goal is to find and rearrange objects as in the instruction “Find the Candle”.\n Here are your observations across time. Timestep 0: You did frontier-based exploration and did an initial scan of the house, before the human moved objects. You made a map of the house in this exploration - we will call this map your “Map 0” (map made at tilmestep 0). \n After timestep 0,.You were standing in the living room (Room 0) and you have partially observed the human moving.\n  Timestep 1: The human moved inside Room 0, holding a candle. The human put the candle from the chair to the countertop.\n Your goal is to find the candle. You should only respond in the format as described below: RESPONSE FORMAT: Reasoning: Based on the information I listed above, do reasoning about what next room to search. Room: Next Room.\n Now, which room should you go to? Please answer with the RESPONSE FORMAT.\n RESPONSE FORMAT:\n Reasoning: Based on the information I listed above, I observed the human move the candle inside Room 0 (the living room) and place it from the chair to the countertop. Therefore, I do not need to search another room since I already know the location of the candle in Room 0.\n Room: Room 0.\n Actual Task: '
		self.baseline_type = baseline_type
		self.magic_grasp = magic_grasp
		self.invalid_receps = set(['bench', 'fridge', 'shelves', 'unidentified_recep'])
		self.human_recep_no_put = human_recep_no_put
		self.call_limit = 15

	def generate_representation(self, input_dict, follow_or_no=False):

		system_prompt = get_system_prompt(input_dict)
		###############################


		#1. Observation
		fbe_map_prompt = "While premapping, you saw the layout of the house; " + self.get_current_map_prompt(input_dict, input_dict['fbe_map_lo_so_dict'], state='premap').replace('While premapping,', '')
		#Intermediate prompt
		if self.baseline_type == 'reasoning':
			interm_prompt = ""
			if len(input_dict['prompt']['intermediate'])> 0:
				interm_prompt = "A little after that, the human said "
				interm_prompt += " A little after that, the human said ".join(["'"+ p['prompt']['sentences'] + "'." for p in input_dict['prompt']['intermediate']])
			if len(interm_prompt) > 0:
				interm_prompt = 'a' + interm_prompt[1:]

			#2. Format
			interm_prompt += " After all of this, the human finally commanded to you: '" + input_dict['prompt']['initial']  + ".' "


			human_recep = input_dict['prompt']['initial_recep_type_human']
			if human_recep:
				human_pose_room_cur = input_dict['human_pose_room_cur']
				human_pose_room_initial = input_dict['human_pose_room_initial']
				human_towards_rooms = input_dict['human_towards_rooms']
				interm_prompt += " Right after the human said this, you saw the human in room " +  str(human_pose_room_initial) + ", starting to walk towards somewhere"
				if input_dict['time'] >1:
					interm_prompt += " When you last saw the human, the human was in room " + str(human_pose_room_cur) 
				if input_dict['human_follow_stopped']:
					interm_prompt += ". You saw that the human has stopped here or either you lost track of the human.\n"
				else:
					if not(human_towards_rooms is None) and len(human_towards_rooms) >0:
						interm_prompt += ", walking towards rooms " + ", ".join([str(h) for h in human_towards_rooms]) + ".\n"
					else:
						interm_prompt += ". \n"


			if self.llm_type == 'llama':
				total_prompt = self.llama_in_contxt_example
			else:
				total_prompt = ""
			if not (input_dict['human_seen_scripted']):
				total_prompt += system_prompt + '\n' + fbe_map_prompt + '\n' + 'Then, ' + interm_prompt + '"\n' 
			else:
				raise Exception("Need to fix this later!")
			total_prompt += self.get_environment_prompts(input_dict)
			map_lo_so_dict = self.most_recent_map_lo_so_dict 
			if input_dict['prompt']['initial_recep_type_human']:
				if input_dict['follow_structure'] == 'follow_first_then_llm':
					if input_dict['human_follow_stopped'] or input_dict['time'] >= 500:
						format_prompt = get_format_prompt(input_dict, self.cur_action_set, map_lo_so_dict, self.cur_pose_room)
					else:
						format_prompt = 'Follow human called, from follow_first_then_llm'
				elif input_dict['follow_structure'] == 'no_follow_llm':
					format_prompt = get_format_prompt(input_dict, self.cur_action_set, map_lo_so_dict, self.cur_pose_room)
				else:
					if follow_or_no and not(input_dict['human_follow_stopped']) and not(input_dict['time'] >= 500):
						format_prompt = get_format_prompt_human_follow(input_dict, self.cur_action_set, map_lo_so_dict, self.cur_pose_room)
					else:
						format_prompt = get_format_prompt(input_dict, self.cur_action_set, map_lo_so_dict, self.cur_pose_room)
			else:
				format_prompt = get_format_prompt(input_dict, self.cur_action_set, map_lo_so_dict, self.cur_pose_room)
			total_prompt += "\n" + format_prompt
		return total_prompt


	def get_current_map_prompt(self, input_dict,map_lo_so_dict_input,  state=None, current_room_only=None):
		map_prompt = map_to_prompt(map_lo_so_dict_input, input_dict, state, current_room_only)
		return map_prompt

	def generate_latest_env_prompt(self, input_dict):
		latest_map_lo_so_dict = self.most_recent_map_lo_so_dict 
		hl_tool_nav=  self.last_return_dict['hl_tool'] in['nav_to_room' or 'explore_room']
		if self.last_return_dict['hl_tool']== 'nav_to_room':
			prompt = "You checked room " + str(self.cur_pose_room) + "; you scanned some parts of it (and did not scan other parts). " 
			prompt += self.get_current_map_prompt(input_dict, latest_map_lo_so_dict, current_room_only =self.cur_pose_room)
		elif self.last_return_dict['hl_tool'] == 'explore_room':
			prompt = "You further explored another location in room " + str(self.cur_pose_room) + ". " 
			prompt += self.get_current_map_prompt(input_dict, latest_map_lo_so_dict, current_room_only =self.cur_pose_room)
		elif self.last_return_dict['hl_tool']== 'grab_obj':
			if not(input_dict['failed_interact']):
				prompt = "You are in room " + str(self.cur_pose_room) + ". You grabbed " + str(self.last_return_dict['ll_argument']['obj']) + "; "  
				self.grabbed_obj = str(self.last_return_dict['ll_argument']['obj'])
			else:
				prompt = "You are in room " + str(self.cur_pose_room) + ". You went in front of the location of " + str(self.last_return_dict['ll_argument']['obj']) + " on your map to grab it; but when you actually went to its location on the map, " + str(self.last_return_dict['ll_argument']['obj'])+ " was not there, and you could not grab it. " 
		elif self.last_return_dict['hl_tool']== 'put_obj':
			if not(input_dict['failed_interact']):
				prompt = "You are in room " + str(self.cur_pose_room) + ". You put the " + str(self.grabbed_obj) + " you were grabbing on " + str(self.last_return_dict['ll_argument']['obj']) + "; " 
				self.grabbed_obj = None
			else: 
				prompt = "You are in room " + str(self.cur_pose_room) + ". You went in front of the location of " + str(self.last_return_dict['ll_argument']['obj']) + " on your map to put the " + str(self.grabbed_obj) + " you are grabbing on it ; but when you actually went to its location on the map, " +  str(self.last_return_dict['ll_argument']['obj'])+" was not there, and you could not put it on the " + str(self.last_return_dict['ll_argument']['obj']) + ". " 
		elif self.last_return_dict['hl_tool']== 'give_human':
			prompt = "You are in room " + str(self.cur_pose_room) + ". You gave the " + str(self.grabbed_obj) + " you were grabbing to the human; " 
			self.grabbed_obj = None
		elif self.last_return_dict['hl_tool'] == "follow_human":
			human_pose_room_cur = input_dict['human_pose_room_cur']
			human_pose_room_initial = input_dict['human_pose_room_initial']
			human_towards_rooms = input_dict['human_towards_rooms']
			prompt = "You were following the human;"
			prompt += "the human was in room " + str(human_pose_room_cur)
			if input_dict['human_follow_stopped']:
				prompt += ". You saw that the human has stopped here or either you lost track of the human.\n"
			else:
				if not(human_towards_rooms is None) and len(human_towards_rooms) >0:
					prompt += ", walking towards rooms " + ", ".join([str(h) for h in human_towards_rooms]) + ".\n"
				else:
					prompt += ". \n"
		elif self.last_return_dict['hl_tool']== "END_TASK":
			prompt = "You ended the task!"
		else:
			raise Exception()
		return prompt

	def get_environment_prompts(self, input_dict):
		prompt = ""
		latest_nav_time = -1

		for k, v in self.environment_prompts.items():
			prompt += "Timestep " + str(k) + " : " + v + "\n"

		prompt += " Combining your latest observations, " + self.get_current_map_prompt(input_dict, self.most_recent_map_lo_so_dict)
		return prompt 

	def generate_current_action_set(self, input_dict):
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
		self.cur_pose_room = cur_pose_room

		explore_room = []
		if self.cur_pose_room != 'free_space':
			explore_room = ['Explore Room '+ str(self.cur_pose_room)]

		go_tos = ["Go to Room " + str(r) for r in input_dict['room_dict']['room_assignment'] if not(r == cur_pose_room or r=='free_space')]
		#Pick up objects
		map_lo_so_dict = self.most_recent_map_lo_so_dict  
		objs = [] 
		if not(cur_pose_room == "free_space"):
			for recep, value_list in map_lo_so_dict[cur_pose_room].items():
				objs += value_list
		human_recep = input_dict['prompt']['initial_recep_type_human']
		if self.human_recep_no_put and human_recep:
			put_allowed = False
		else:
			put_allowed = True
		manipulation_objs =  []
		if not(cur_pose_room == "free_space"):
			if not(input_dict['is_holding'] == None) and put_allowed: #put_objs 
				manipulation_objs = ["Put " + input_dict['is_holding'] + " on the " +recep for recep in map_lo_so_dict[cur_pose_room] if (recep!='unidentified_recep' and recep!='human' and not(recep in self.invalid_receps))]
			elif input_dict['is_holding'] == None: #pick_ups 
				manipulation_objs = ["Pick up " + obj for obj in objs]

		give_to_human = []
		if human_recep and not(cur_pose_room == "free_space") and (cur_pose_room == input_dict['human_walking_to_room']) and not(input_dict['is_holding'] == None) and not(self.grabbed_obj == None):
			give_to_human = ["Give " + input_dict['is_holding'] + " to the " + "human"]
		return explore_room + go_tos + manipulation_objs + give_to_human #+ follow_the_human


	def decide_generate_new_representation(self, input_dict):
		if input_dict['new_agent_status'] or input_dict['enviornment change']:
			return True
		else:
			return False


	def decide_whether_execute(self, input_dict):
		task_end_conidition = input_dict['time'] >= self.end_time
		if input_dict['last_tool_ended'] == True or task_end_conidition:
			return True
		else:
			return False 




	###########################
	def choose_low_level(self, high_level_tool, input_dict):
		if high_level_tool == "nav_to_room":
			return llm_choose_room(llm_representation)

		elif high_level_tool == "explore_room":
			raise Exception("Not Implemented!")

		elif high_level_tool == "explore_room":
			raise Exception("Not Implemented!")


	###########################
	def reconstruct_action_output(self, input_dict, action_output):
		action_output = ''.join(c for c in action_output if c not in string.punctuation.replace('_', '').replace('!', ''))
		try:
			if 'Pick up ' in action_output:
				#Then return pickup tool
				obj_entity = action_output.replace('Pick up ', '')
				reconstruct_action_output = 'Pick up ' + obj_entity
			elif "Put " in action_output:
				recep = action_output.split("on the ")[-1].replace(".", "")
				reconstruct_action_output = 'Put '+ input_dict['is_holding'] + " on the " +recep 
			elif 'Go to ' in action_output:
				room_number = int(action_output.replace('Go to ', '').replace('Room ', '')) #Room number
				reconstruct_action_output = "Go to Room "+ str(room_number)
			elif 'Explore Room ' in action_output:
				room_number = int(action_output.replace('Explore Room ', '').replace('Room ', '')) #Room number
				reconstruct_action_output = "Explore Room "+ str(room_number)
			elif "Give " in action_output:
				obj_entity = action_output.replace('Give ', '').replace(' to the human', '')
				reconstruct_action_output = "Give " + obj_entity + " to the " + "human"
				#breakpoint()
			elif "Not Enough Evidence" in action_output:
				reconstruct_action_output = "Follow the Human"
			elif "Enough Evidence" in action_output:
				reconstruct_action_output = 'Complete the task without following the human'
			else:
				reconstruct_action_output = action_output
		except:
			reconstruct_action_output = action_output

		return reconstruct_action_output



	###########################
	def execute(self, input_dict):
		#Reset environment prompts
		if self.llm_type=='openai':
			self.reasoner =  instantiate_llm('openai_chat', generation_params={'model':'gpt-3.5-turbo-0125'})
		elif self.llm_type == 'openai_chat':
			self.reasoner =  instantiate_llm(self.llm_type, generation_params={'model':'gpt-4o'})
		if input_dict['time']==0:
			self.environment_prompts = {}
			self.grabbed_obj = None
			self.enough_evidence = False
			self.last_return_dict = {}
			self.num_called = 0 
		

		if input_dict['ignore_cur_ep']:
			return {'hl_tool': 'END_TASK', 'whether_execute': True}

		if input_dict['prompt']['initial_recep_type_human'] and input_dict['time']==0:
			hl_tool = 'no_op'
			room_number = 0 
			ask_again_every_20 = True 
			ll_arg = {'ask_again_every_20': ask_again_every_20}
			self.last_return_dict = {'hl_tool': hl_tool, 'll_argument_before_adjust': ll_arg, 'whether_execute': True,
										'llm_rep': None,
										'gen_string': None,
										'action_output': None,
										'reasoning_output': None,
										'chosen_room_number': room_number}
			return self.last_return_dict


		if input_dict['task_phase']:
			whether_execute = self.decide_whether_execute(input_dict)
			whether_new_rep = self.decide_generate_new_representation(input_dict)
			if whether_new_rep or whether_execute:
				self.most_recent_map_lo_so_dict = get_map_lo_so_dict(input_dict, include_human=False) 
				self.cur_action_set = self.generate_current_action_set(input_dict) + ["Done!"] 
				if self.last_return_dict != {} and self.last_return_dict['hl_tool']!='no_op':
					self.environment_prompts[input_dict['time']] = self.generate_latest_env_prompt(input_dict)
				if (input_dict['time'] >= self.end_time) or  self.num_called >=  self.call_limit: 
					#breakpoint()
					return {'hl_tool': 'END_TASK', 'whether_execute': True}
					 
				complete_without_following = False

				llm_rep = self.generate_representation(input_dict, follow_or_no=not(self.enough_evidence))

				try:
					gen_string = self.reasoner.generate(llm_rep)
				except Exception as e:
					time.sleep(60)
					try:
						gen_string = self.reasoner.generate(llm_rep)
					except Exception as e:
						gen_string = str(e)

				action_output = gen_string.split('Action: ')[-1]
				reasoning_output = gen_string.split('Action: ')[0].split("Reasoning: ")[-1]
				action_output = self.reconstruct_action_output(input_dict, action_output)
				
				print('llm rep before: ', llm_rep)
				print('gen string before: ', gen_string)

				if action_output == 'Complete the task without following the human':
					gen_string_evidence = copy.deepcopy(gen_string)
					reasoning_output_evidence = copy.deepcopy(reasoning_output)
					action_output_evidence = copy.deepcopy(action_output)

					self.enough_evidence = True
					if self.llm_type=='openai':
						self.reasoner =  instantiate_llm('openai_chat', generation_params={'model':'gpt-3.5-turbo-0125'})
					elif self.llm_type == 'openai_chat':
						self.reasoner =  instantiate_llm(self.llm_type, generation_params={'model':'gpt-4o'})

					complete_without_following = True
					#Then ask again 
					llm_rep = self.generate_representation(input_dict, follow_or_no=False)
					try:
						gen_string = self.reasoner.generate(llm_rep)
					except Exception as e:
						time.sleep(60)
						try:
							gen_string = self.reasoner.generate(llm_rep)
						except Exception as e:
							gen_string = str(e)
					action_output = gen_string.split('Action: ')[-1]
					reasoning_output = gen_string.split('Action: ')[0].split("Reasoning: ")[-1]
					action_output = self.reconstruct_action_output(input_dict, action_output)

					gen_string = gen_string_evidence + "\n" + gen_string




				if action_output in self.cur_action_set + ['Follow the Human']:
					if not('Follow the Human' in action_output):
						self.num_called +=1

					if 'Pick up ' in action_output:
						hl_tool = "grab_obj"
						obj_entity = action_output.replace('Pick up ', '')
						ll_arg = {'obj': obj_entity, 'in_room': input_dict['room_dict']['room_assignment'][self.cur_pose_room][input_dict['lmb'][0]:input_dict['lmb'][1], input_dict['lmb'][2]:input_dict['lmb'][3]] }
						room_number = self.cur_pose_room
					elif "Put " in action_output:
						hl_tool = "put_obj"
						recep = action_output.split("on the ")[-1].replace(".", "")
						ll_arg = {'obj': recep, 'in_room': input_dict['room_dict']['room_assignment'][self.cur_pose_room][input_dict['lmb'][0]:input_dict['lmb'][1], input_dict['lmb'][2]:input_dict['lmb'][3]] }
						room_number = self.cur_pose_room
					elif "Give " in action_output:
						hl_tool = "give_human"
						room_number = self.cur_pose_room
						ll_arg = {'hl_tool': hl_tool, 'whether_execute': True, 'chosen_room_number': self.cur_pose_room, 'llm_rep': llm_rep, 'gen_string': gen_string}
					elif 'Go to ' in action_output:
						hl_tool = 'nav_to_room'
						room_number = int(action_output.replace('Go to ', '').replace('Room ', '')) #Room number
						ll_arg = room_goal(input_dict['time'], input_dict['room_dict']['room_assignment'], room_number) #select_random_coord_in_room(input_dict['time'], input_dict['room_dict']['room_assignment'], room_number) #
					elif 'Explore Room ' in action_output:
						hl_tool = 'explore_room'
						room_number = int(action_output.replace('Explore ', '').replace('Room ', '')) #Room number
						ll_arg = room_goal(input_dict['time'], input_dict['room_dict']['room_assignment'], room_number, explore_room=True)
					elif 'Follow the Human' in action_output:
						hl_tool = 'follow_human'
						room_number = 0 #Just do 0 
						ask_again_every_20 = not(self.enough_evidence ) and not(input_dict['human_follow_stopped']) and not(input_dict['time'] >= 500) and not(input_dict['follow_structure'] in ['follow_first_then_llm', 'no_follow_llm'])
						ll_arg = {'ask_again_every_20': ask_again_every_20}
					elif 'Done' in action_output:
						room_number = "Done declared!"
						return {'hl_tool': 'END_TASK', 'whether_execute': True, 'chosen_room_number':room_number, 'llm_rep': llm_rep, 'gen_string': gen_string}
					else:
						room_number = "couldn't parse action output"
						return {'hl_tool': 'END_TASK', 'whether_execute': True, 'chosen_room_number':room_number, 'llm_rep': llm_rep, 'gen_string': gen_string}
				else:
					room_number = "couldn't parse action output"
					return {'hl_tool': 'END_TASK', 'whether_execute': True, 'chosen_room_number':room_number, 'llm_rep': llm_rep, 'gen_string': gen_string}
				if complete_without_following:
					llm_rep = "You had chosen to Complete the task without following the human!\n" + llm_rep

				
				print("LLM gen string ", gen_string)
				print("Chose room ", room_number, "!")
				self.last_return_dict = {'hl_tool': hl_tool, 'll_argument_before_adjust': ll_arg, 'whether_execute': True,
										'llm_rep': llm_rep,
										'gen_string': gen_string,
										'action_output': action_output,
										'reasoning_output': reasoning_output,
										'chosen_room_number': room_number}
			#Adjust 
			if self.last_return_dict['hl_tool'] in ["nav_to_room", "explore_room"]:
				self.last_return_dict['ll_argument'] = self.last_return_dict['ll_argument_before_adjust'] 
			else:
				self.last_return_dict['ll_argument'] = self.last_return_dict['ll_argument_before_adjust']
			self.last_return_dict['whether_execute'] = whether_new_rep or whether_execute
			return self.last_return_dict
		else:
			self.last_return_dict = {}
			return self.last_return_dict 


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
                                large_objects_connected_regions[l][room_num] = np.zeros(room_map.shape) 
                            large_objects_connected_regions[l][room_num] += (connected_regions==c)
                            large_objects_connected_regions[l][room_num] = 1.0 *(large_objects_connected_regions[l][room_num]>0)
    return large_objects_connected_regions


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
	
	prompt = 'The human said your goal is to “' + input_dict['prompt']['initial'] + '”' + '. ' + 'The human may be nuanced or ambiguous.' 
	human_recep = input_dict['prompt']['initial_recep_type_human']
	if human_recep:
		human_pose_room_cur = input_dict['human_pose_room_cur']
		human_pose_room_initial = input_dict['human_pose_room_initial']
		human_towards_rooms = input_dict['human_towards_rooms']
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
	prompt +="Please ONLY respond in the format: RESPONSE FORMAT: Reasoning: reason about 'Enough Evidence'/ 'Not Enough Evidence'. Action: choose among ['Enough Evidence', 'Not Enough Evidence']. RESPONSE FORMAT:\n"
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
		
	prompt += " To achieve the goal, choose the next action among Available Actions: "+ str(cur_action_set) 
	if human_recep:
		prompt += ' If "Give x to the human" was in the Available Actions, it means the human is in the same room as you, and you can hand the object you have to the human. If you were told to "bring something to the human", please remember that you can ONLY succeed the task if you choose "Give x to the human", where x is an object, as the final action. You can only choose  "Give x to the human" when it is available.\n '
	else:
		prompt += "' .\n "
	prompt += "You are in room " + str(cur_pose_room) + " and you can ONLY grab/ put objects inside Room " + str(cur_pose_room) #". These objects are: " + str(list(map_lo_so_dict[cur_pose_room].keys()) + objects_in_room)  + ", since they are in room " + str(cur_pose_room) + " . You CANNOT put to/ grab other objects.\n"
	if str(cur_pose_room != 'free_space'):
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
		selem = skimage.morphology.disk(20) 
		goal = skimage.morphology.binary_erosion(goal, selem)
		if np.sum(goal) ==0:
			goal_dilated= room_dict[room] * 1.0
			selem = skimage.morphology.disk(1)
			while np.sum(goal_dilated) >10:
				goal = copy.deepcopy(goal_dilated)
				goal_dilated = skimage.morphology.binary_erosion(goal, selem)
	return goal*1.0

def select_random_coord_in_room(seed, room_dict, room):
	np.random.seed(seed)
	wheres = np.where(room_dict[room])
	len_where = len(wheres[0])
	np.random.seed(seed)
	where_idxes = np.random.choice(np.arange(len_where), 10)
	global_goals = [[wheres[0][where_idx], wheres[1][where_idx]] for where_idx in where_idxes]
	return global_goals