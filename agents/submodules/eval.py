import os
import numpy as np
import pickle

class Eval:
	def __init__(self, agent):
		self.agent = agent

	
	def get_object_translations_cat(self, cat_name): #e.g. cat_name is 'hat'
		rom = self.agent._env._sim.get_rigid_object_manager()
		cur_obj_pose = {handle: rom.get_object_by_handle(handle).translation for handle in self.agent.perception.all_the_handles} 
		obj_tranaslations = []
		obj_rotations = []
		obj_handles = []
		for k, v in cur_obj_pose.items():
			if self.agent.perception.handles_to_catergories[k.split('_:')[0]] == cat_name:
				obj_handles.append(k)
				obj_tranaslations.append(v)
				obj_rotations.append(rom.get_object_by_handle(k).rotation)
		return obj_handles, obj_tranaslations, obj_rotations


	def save_model_task_run_output(self, write=False):
		goal_obj, goal_recep, task_type =  self.agent.reset_task.get_goal_cat_and_recep()
		grasp_mgr = self.agent.habitat_env.sim.get_agent_data(0).grasp_mgr
		grasper_closed = not(grasp_mgr.is_grasped)
		#
		recep_type_dict = None
		room_type_dict = None
		human_type_dict = None
		#
		if task_type == 'recep':
			possible_recep_insts = self.agent.get_valid_recep_instance(goal_recep)
			obj_handles_changed, obj_translations_changed, obj_rotations_changed = self.get_object_translations_cat(goal_obj)
			recep_type_dict = {'possible_recep_insts': [p.name for p in possible_recep_insts],
									'obj_handles_changed': obj_handles_changed,
									'obj_translations_changed': obj_translations_changed,
									'obj_rotations_changed': obj_rotations_changed}
		#
		elif task_type == 'room':
			obj_handles_changed, obj_translations_changed, obj_rotations_changed = self.get_object_translations_cat(goal_obj)
			room_type_dict = {'obj_handles_changed': obj_handles_changed,
									'obj_translations_changed': obj_translations_changed,
									'obj_rotations_changed': obj_rotations_changed}
		#
		else:
			obj_handles_changed, obj_translations_changed, obj_rotations_changed = self.get_object_translations_cat(goal_obj)
			human_grasper = self.agent.habitat_env.sim.get_agent_data(1).grasp_mgr
			human_grasper_grasped = human_grasper.is_grasped

			#object pose close to human pose
			#human_base_pos 
			human_base_pos = np.array(self.agent._env._sim.get_agent_data(1).articulated_agent.sim_obj.translation)[[0,2]] 
			human_type_dict = {'obj_handles_changed': obj_handles_changed,
									'obj_translations_changed': obj_translations_changed,
									'obj_rotations_changed': obj_rotations_changed,
									'human_grasper_grasped': human_grasper_grasped,
									'human_base_pos': human_base_pos,
									'human_agent_distance': self.agent.human_agent_distance,
									'agent_which_room': self.agent.agent_which_room}
		#
		task_type_dict = {'recep_type_dict': recep_type_dict,
									'room_type_dict': room_type_dict,
									'human_type_dict': human_type_dict}
		#       
		dict_to_save = {'goal_obj': goal_obj,
						'goal_recep': goal_recep, 
						'task_type': task_type,
						'grasper_closed': grasper_closed, 
						'task_type_dict': task_type_dict,
						'steps_taken': self.agent.timestep} #robot grasper

		if write:
			args = self.agent.args
			dump_dir = "{}/dump/{}/".format(self.agent.args.dump_location,self.agent.args.exp_name)
			pickle_file_path = '{}/episodes/thread_{}/eps_{}/model_output.p'.format(dump_dir, self.agent.rank, self.agent.episode_id)
			pickle_file_folder = os.path.dirname(pickle_file_path)
			if not(os.path.exists(pickle_file_folder)):
				os.makedirs(picklefile_folder)
			pickle.dump(dict_to_save, open(pickle_file_path, 'wb'))

		#
		return dict_to_save

	def get_first_room(self, time2roomdict):
		sorted_keys = sorted(list(time2roomdict.keys()))
		initial_room = time2roomdict[1]
		for k in sorted_keys:
			if not(time2roomdict[k] in ['free space' in 'free_space']) and time2roomdict[k] != initial_room:
				break
		return time2roomdict[k]


	def evaluate_model(self, model_output, write=False): #, goal_obj, goal_recep):
		goal_obj, goal_recep, task_type = model_output['goal_obj'], model_output['goal_recep'], model_output['task_type']
		grasper_closed = model_output['grasper_closed']
		task_type_dict = model_output['task_type_dict']
		succ = False

		
		
		if task_type in ['room', 'recep']:
			if task_type == 'recep':
				possible_recep_insts_names, obj_handles_changed, obj_translations_changed = \
				task_type_dict['recep_type_dict']['possible_recep_insts'], task_type_dict['recep_type_dict']['obj_handles_changed'], task_type_dict['recep_type_dict']['obj_translations_changed']
				#retrieve possible recep insts
				possible_recep_insts = [self.agent.receptacle_instances[p_name] for p_name in possible_recep_insts_names]
				for obj_translation in obj_translations_changed:
					for recep_instance in possible_recep_insts:
						if not(succ):
							succ = self.agent.retr.obj_on_recep(obj_translation, recep_instance)

				
			if task_type=='room':
				room_function = goal_recep
				obj_handles_changed, obj_translations_changed, obj_rotations_changed = \
				task_type_dict['room_type_dict']['obj_handles_changed'], task_type_dict['room_type_dict']['obj_translations_changed'], task_type_dict['room_type_dict']['obj_rotations_changed']

				for obj_translation, obj_rotation in zip(obj_translations_changed, obj_rotations_changed):
					if not(succ):
						succ = self.agent.retr.obj_in_room(obj_translation, obj_rotation, room_function, grasper_closed)
					
		#If task was s_hum
		#Check that the state of the object is changed to given to human
		else:
			#check that the robot is in the same room as human  
			obj_handles_changed, obj_translations_changed, obj_rotations_changed = \
			task_type_dict['human_type_dict']['obj_handles_changed'], task_type_dict['human_type_dict']['obj_translations_changed'], task_type_dict['human_type_dict']['obj_rotations_changed']

			human_grasper_grasped = task_type_dict['human_type_dict']['human_grasper_grasped']

			human_base_pos = task_type_dict['human_type_dict']['human_base_pos']
			for obj_translation in obj_translations_changed:
				if not(succ):
					succ = human_grasper_grasped and (np.linalg.norm(np.array(obj_translation)[[0,2]]  - human_base_pos , ord=2) < 0.5)
					if not(self.agent.args.oracle_baseline_follow_first):
						clear = not(self.agent._env._sim.get_agent_data(0).articulated_agent._sim.ep_info.sif_params['ambiguous'])
						if clear:
							first_room = self.get_first_room(task_type_dict['human_type_dict']['agent_which_room'])
							human_dest_room = self.agent._env._sim.get_agent_data(0).articulated_agent._sim.ep_info.sif_params['prompt']['initial']['walking_to_room']
							succ = succ and not(human_dest_room == first_room)
		if write:
			args = self.agent.args
			dump_dir = "{}/dump/{}/".format(self.agent.args.dump_location,self.agent.args.exp_name)
			pickle_file_path = '{}/episodes/thread_{}/eps_{}/eval_result.p'.format(dump_dir, self.agent.rank, self.agent.episode_id)
			pickle_file_folder = os.path.dirname(pickle_file_path)
			if not(os.path.exists(pickle_file_folder)):
				os.makedirs(picklefile_folder)
			pickle.dump(succ, open(pickle_file_path, 'wb'))
		return succ
