import os
import torch
import numpy as np
import skimage.morphology
import copy

class Util:
	def __init__(self,  agent):
		self.agent = agent


	def are_last_five_consecutive(self, numbers):
		# Check if the list has at least five elements
		if len(numbers) < 5:
			return False

		# Slice the list to get the last five elements
		last_five = numbers[-5:]

		# Check if each number in the last five is one greater than the number before it
		return all(last_five[i] - last_five[i-1] == 1 for i in range(1, 5))


	def get_top_down_distance(self, vector1, vector2):
		return np.linalg.norm(np.array(vector1)[[0, 2]] - np.array(vector2)[[0, 2]])


	def get_map_lo_so_dict(self, input_dict, include_human):
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


	def print_log_text(self, planner_inputs):
		self.agent.print_log("Step: ", self.agent.timestep)
		if self.agent.args.task_phase:
			if self.agent.timestep == 0:
				self.agent.print_log("fbe map lo so dict was ", self.agent.info['fbe_map_lo_so_dict'])
				self.agent.print_log("scripted actions map lo so dict was ", self.agent.info['scripted_map_lo_so_dict'])
		if self.agent.args.task_phase and torch.cuda.is_available():
			if planner_inputs['cur_tools']['whether_execute']:
				self.agent.num_policy_called +=1
				self.agent.info['execute_count'] = self.agent.num_policy_called
				self.agent.policy_called_timesteps.append(self.agent.timestep)
				self.agent.list_of_cur_tools.append(planner_inputs['cur_tools'])
				if self.agent.args.oracle_baseline or self.agent.args.follow_human_baseline:
					self.agent.print_log("fbe map lo so dict was ", self.agent.info['fbe_map_lo_so_dict'])
					self.agent.print_log(self.agent.info['ep_prompt'])

					self.agent.print_log('==================================================================================================')
					self.agent.print_log('Hl Tool is ', planner_inputs['cur_tools']['hl_tool'])
					if planner_inputs['cur_tools']['hl_tool'] != "END_TASK":
						if 'chosen_room_number' in planner_inputs['cur_tools']:
							self.agent.print_log("Chosen room number is ", planner_inputs['cur_tools']['chosen_room_number'])
					self.agent.print_log('==================================================================================================')

				else:
					if planner_inputs['cur_tools']['hl_tool'] == "END_TASK":
						self.agent.print_log('Hl Tool is ', planner_inputs['cur_tools']['hl_tool'])
						if 'chosen_room_number' in planner_inputs['cur_tools']:
							self.agent.print_log(planner_inputs['cur_tools']['chosen_room_number'])
							self.agent.print_log('LLM Rep is ', planner_inputs['cur_tools']['llm_rep'])
							self.agent.print_log('Generated Output is ', planner_inputs['cur_tools']['gen_string'])
					else:
						self.agent.print_log('Hl Tool is ', planner_inputs['cur_tools']['hl_tool'])
						self.agent.print_log('LLM Rep is ', planner_inputs['cur_tools']['llm_rep'])
						self.agent.print_log('Generated Output is ', planner_inputs['cur_tools']['gen_string'])
						self.agent.print_log('Chosen Room Number is ', planner_inputs['cur_tools']['chosen_room_number'])



	def _write_log(self):
		args = self.agent.args
		dump_dir = "{}/dump/{}/".format(self.agent.args.dump_location,self.agent.args.exp_name)
		log_file_path = '{}/episodes/thread_{}/eps_{}/log.txt'.format(dump_dir, self.agent.rank, self.agent.episode_id)
		log_file_folder = os.path.dirname(log_file_path)
		if not(os.path.exists(log_file_folder)):
			os.makedirs(log_file_folder)

		#Write
		f = open(log_file_path, 'w')
		f.write("===================================================\n") 
		for log in self.agent.logs:
			f.write(log + "\n")
		f.close()
