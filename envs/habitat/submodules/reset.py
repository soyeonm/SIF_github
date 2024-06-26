import os
import pickle
import numpy as np
import magnum as mn


from constants import agent_action_prefix, grabbable_object_categories, large_objects, small_objects, categories_to_include


class ResetTask:
	def __init__(self, agent):
		self.agent = agent


	#For semantic segmentation
	def save_or_pass_generate(self):
		self.agent.human_walk_dest = None
		self.agent.info['pddl_list'] = self.read_pddl_seq()


	def read_pddl_seq(self):
		args = self.agent.args
		dump_dir = os.path.dirname(self.agent.config['dataset']['data_path']) + '/oracle_pddl/' + self.agent.config['dataset']['data_path'].split('/')[-1].replace('.json.gz', '').replace('_step1_filter', '')
		pickle_file_path = dump_dir + '/' + self.agent.episode_id + '.p'
		pddl = pickle.load(open(pickle_file_path, 'rb'))
		if self.agent.args.oracle_baseline_follow_first:
			walking_to_room = self.agent._env._sim.get_agent_data(0).articulated_agent._sim.ep_info.sif_params['prompt']['initial']['walking_to_room']
			pddl = [{'hl_tool': 'follow_human', 'll_argument': {'ask_again_every_20': False}, 'chosen_room_number': walking_to_room}] + pddl



		return pddl


	def reset_pose_phase2(self):
		#for task phase after exploration
		self.recalibrate_with_map_center()
		self.teleport_with_sif_params_and_recalibrate()
		self.load_sif_obj_params()
		self.agent.human_agent_distance = {}
		self.agent.agent_which_room = {}

	def reset_info(self, phase1=True):
		self.agent.rom = self.agent._env.sim.get_rigid_object_manager()
		self.agent.info['put_log'] = []
		self.agent.last_cam_down = False
		self.agent.info['intermediate_stop_human_follow'] = False
		self.agent.info['failed_interact'] = None

		self.agent.info['human_towards_rooms'] = None
		self.agent.info['human_pose_room_cur'] = None 
		self.agent.info['human_pose_room_initial'] = None
		self.agent.info['human_follow_stopped']  = False 
		self.agent.prev_human_follow_ona = False

		self.agent.timestep = 0
		self.agent.counter_stop_follow = 0
		self.agent.human_traj_at_ona =  None
		self.agent.just_reset = True


		self.agent.info['time'] = self.agent.timestep
		self.agent.info['success'] = None
		self.agent.info['sensor_pose'] = [0., 0., 0.]
		self.agent.info['episode_json'] = self.agent.config['dataset']['data_path'].split('/')[-1].replace('.json.gz', '')
		self.agent.info['jsons_dir'] =  self.agent.config['dataset']['data_path'].split('/jsons/')[0]
		self.agent.info['scripted_map_lo_so_dict'] = None
		self.agent.info['human_walking_to_room'] = None
		self.agent.info['pddl_list'] = None
		self.agent.info['reset_phase2'] = False
		self.agent.info['failed_human_spawn'] = False



		self.agent.grab_log = []
		self.agent.put_log = []

		self.agent.num_human_observed = 0

		self.agent.human_seen_scripted = False
		self.agent.turn_count =0

		# Initializations
		
		self.agent.stopped = False
		self.agent.path_length = 1e-5


		#Agent
		self.agent._env._sim.get_agent_data(0).articulated_agent.at_goal_ona = False


		if phase1:
			self.agent.episode_no += 1
			self.agent.episode_id = self.agent._env._sim.get_agent_data(0).articulated_agent._sim.ep_info.episode_id
			self.agent.info['ignore_cur_ep'] = (self.agent.episode_id in self.agent.eps_saved_in_tmp_dump) or (not(self.agent.eps_to_run is None) and not(self.agent.episode_id in self.agent.eps_to_run))
			self.agent.ep_info = self.agent._env._sim.get_agent_data(0).articulated_agent._sim.ep_info   
			self.agent.info['episode_id'] = self.agent.episode_id

			self.agent.print_log("episode id is ", self.agent.episode_id)
			self.agent.print_log("ep info is ", self.agent._env._sim.get_agent_data(0).articulated_agent._sim.ep_info)
			self.agent.info['fbe_map_lo_so_dict'] = None


		if self.agent.args.run_full_task and phase1:
			self.agent.args.replay_fbe_actions_phase = True
			self.agent.args.task_phase = False
			self.agent.args.replay_scripted_actions_phase = False
			self.agent.print_log("starting full task with replay_fbe_actions_phase")

	#Just temporarily here
	def reset_episode_json(self):
		load_from_json_name = self.agent._env._sim.get_agent_data(0).articulated_agent._sim.ep_info.load_from_json.split('/')[-1].replace('.json.gz', '')
		load_from_ep_idx = str(self.agent._env._sim.get_agent_data(0).articulated_agent._sim.ep_info.load_from_json_ep_idx)
		self.agent.info['load_from_json_name'] = load_from_json_name
		self.agent.info['load_from_ep_idx'] = load_from_ep_idx

	def reset_sif_prompt(self):
		prompt_to_pass = {'initial': self.agent._env._sim.get_agent_data(0).articulated_agent._sim.ep_info.sif_params['prompt']['initial']['prompt'],
									'intermediate': self.agent._env._sim.get_agent_data(0).articulated_agent._sim.ep_info.sif_params['prompt']['intermediate'],
									'initial_recep_type_human': self.agent.ep_info.sif_params['prompt']['recep_type']=='human'}
		return prompt_to_pass


	def reset_info_obj(self):
		self.agent.info['small_obj_channels'] = [categories_to_include[small_objects[0]], categories_to_include[small_objects[-1]]]  
		self.agent.info['large_objects'] = {categories_to_include[l]:l for l in large_objects}
		self.agent.info['small_objects'] = {categories_to_include[s]:s for s in small_objects} 

	def reset_pose(self):
		if not(self.agent.args.premapping_fbe_mode):
			fbe_repl_starting_pos_dict = pickle.load(open(os.path.join(self.agent.fbe_actions_dir, 'starting_pose.p'), 'rb')) 
			self.agent.fbe_repl_visited_pos_list = pickle.load(open(os.path.join(self.agent.fbe_actions_dir, 'visited_pos.p'), 'rb'))
			

		if self.agent.args.replay_fbe_actions_phase:
			self.agent._env._sim.agents_mgr[0].articulated_agent.sim_obj.transformation = fbe_repl_starting_pos_dict['agent0_trans'] 
			self.agent._env._sim.agents_mgr[1].articulated_agent.sim_obj.transformation = fbe_repl_starting_pos_dict['agent1_trans'] 


		self.agent.loc.reset(self.agent._env, self.agent.args, starting=True)
		self.agent.info['camera_height'] = self.agent._env._sim.get_agent_data(0).articulated_agent._sim._sensors["agent_0_articulated_agent_arm_depth"]._sensor_object.node.transformation.translation[1] #self.agent.content_scenes

	def reset_info_for_task_phase(self):
		self.agent.args.replay_fbe_actions_phase = False
		self.agent.args.task_phase = True
		self.agent.timestep = 0
		self.agent.info['task_phase'] = self.agent.args.task_phase
		self.agent.info['intermediate_stop'] = True 
		self.agent.info['time'] = 0
		self.agent.stg = None
		self.agent.prev_fmm = None
		self.agent._env._sim.get_agent_data(0).articulated_agent.at_goal_ona = False



	def reset_prompt_and_room_dict(self):
		if self.agent.args.replay_fbe_actions_phase:
			self.agent.prompt = self.agent._env._sim.get_agent_data(0).articulated_agent._sim.ep_info.sif_params['prompt']
			json_gz_path = self.agent.config['dataset']['data_path']
			root_data_folder = os.path.dirname(self.agent.config['dataset']['data_path']).replace('/jsons', '')
			scene_id = self.agent._env._sim.get_agent_data(0).articulated_agent._sim.ep_info.scene_id.split('/')[-1].replace('.scene_instance.json', '')
			self.agent.fbe_actions_dir = os.path.join('fbe_mapping_actions', 'official' , scene_id)
			room_data_dir = os.path.join(root_data_folder, 'room_annotations', scene_id)
			self.agent.info['room_dict'] = pickle.load(open(os.path.join(room_data_dir, 'room_annotation_dict.p'), 'rb'))
			self.agent.retr.reset(self.agent.info['room_dict'])
			self.agent.perception.viz.reset(None)
		else:
			self.agent.prompt = self.agent._env._sim.get_agent_data(0).articulated_agent._sim.ep_info.sif_params['prompt']
			json_gz_path = self.agent.config['dataset']['data_path']
			root_data_folder = os.path.dirname(self.agent.config['dataset']['data_path']).replace('/jsons', '')
			scene_id = self.agent._env._sim.get_agent_data(0).articulated_agent._sim.ep_info.scene_id.split('/')[-1].replace('.scene_instance.json', '')
			self.agent.fbe_actions_dir = os.path.join('fbe_mapping_actions', 'official' , scene_id)
			self.agent.perception.viz.reset(self.agent.prompt['initial']['prompt'])


	def reset_sif_task_info(self):
		self.agent.info['task_phase'] = self.agent.args.task_phase
		self.agent.info['holding'] = None
		self.agent.list_of_cur_tools = []
		self.agent.grabbed_obj_id = None
		self.agent.info['human_seen_scripted'] = self.agent.human_seen_scripted
		self.agent.num_policy_called = 0
		self.agent.policy_called_timesteps = []
		self.agent.info['execute_count'] = self.agent.num_policy_called


	def recalibrate_with_map_center(self):
		fbe_repl_starting_pos_dict = pickle.load(open(os.path.join(self.agent.fbe_actions_dir, 'starting_pose.p'), 'rb')) 
		self.agent._env._sim.agents_mgr[0].articulated_agent.sim_obj.transformation = fbe_repl_starting_pos_dict['agent0_trans'] 
		self.agent._env._sim.maybe_update_articulated_agent()
		self.agent.habitat_env.sim.internal_step(-1)
		
		self.agent.loc.agent_0_starting_camera_pose = self.agent.loc.get_sim_location() 
		self.agent.loc.agent_0_starting_agent_pose =  self.agent.loc.get_sim_agent_location() 
		self.agent.loc.rel_o_offset = (self.agent.loc.get_sim_agent_location()[2] - self.agent.loc.get_sim_location()[2]-np.pi)*180/np.pi

		self.agent.loc.agent_0_starting_camera_pose_in_agent_coordinate = (self.agent.loc.get_ee_cam_location()[0], self.agent.loc.get_ee_cam_location()[1], self.agent.loc.get_ee_cam_location()[2]) 


	def teleport_with_sif_params_and_recalibrate(self):
		#0. AGENT Agent pose 
		premap_load_folder = 'task_load_folder'
		save_folder = os.path.join(premap_load_folder, self.agent.info['jsons_dir'].split('/')[-1], self.agent.info['load_from_json_name'], self.agent.info['load_from_ep_idx'])
		pose_dict = pickle.load(open(os.path.join(save_folder, 'local_pose.p'), 'rb'))
		self.agent._env._sim.agents_mgr[0].articulated_agent.sim_obj.transformation =  pose_dict['habitat_pose'] 
		#SIF
		self.agent._env._sim.maybe_update_articulated_agent()
		self.agent.habitat_env.sim.internal_step(-1)

		self.agent.loc.reset(self.agent._env, self.agent.args, starting=False)

		#SIF
		self.agent._env._sim.agents_mgr[0].articulated_agent.sim_obj.transformation = mn.Matrix4(np.array(self.agent._env._sim.get_agent_data(0).articulated_agent._sim.ep_info.sif_params['agent_start_pose']))   
		self.agent._env._sim.maybe_update_articulated_agent()
		self.agent.habitat_env.sim.internal_step(-1)

		dx, dy, do = self.agent.loc.get_pose_change()
		self.agent.info['sensor_pose'] = [dx, dy, do]


		#1. Human pose
		if self.agent.task_type!='human':
			self.agent._env._sim.agents_mgr[1].articulated_agent.sim_obj.transformation = mn.Matrix4(np.array(self.agent._env._sim.get_agent_data(0).articulated_agent._sim.ep_info.sif_params['human_start_pose']))
			self.agent.human_walk_dest = None
			self.agent.walking_to_room = None

		else:
			self.agent.walking_to_room = self.agent._env._sim.get_agent_data(0).articulated_agent._sim.ep_info.sif_params['prompt']['initial']['walking_to_room']
			self.agent._env._sim.agents_mgr[1].articulated_agent.sim_obj.transformation = mn.Matrix4(np.array(self.agent._env._sim.get_agent_data(0).articulated_agent._sim.ep_info.sif_params['human_start_pose']))
			self.agent.human_walk_dest = np.array(self.agent._env._sim.get_agent_data(0).articulated_agent._sim.ep_info.sif_params['human_dest_pose'])
			self.agent.initial_human_pose_habitat =self.agent._env._sim.agents_mgr[1].articulated_agent.sim_obj.transformation

		self.agent.info['human_walking_to_room'] = self.agent.walking_to_room

		curr_pose_start_x, curr_pose_start_y, curr_pose_start_o = pose_dict['start_startx_starty']['start_x'], pose_dict['start_startx_starty']['start_y'] , pose_dict['pose_pred'][2]
		self.agent.start_x, self.agent.start_y = curr_pose_start_x, curr_pose_start_y
		self.agent.curr_loc = [curr_pose_start_x, curr_pose_start_y, curr_pose_start_o]

		#2. Finally 
		#Adjust camera height
		self.agent.info['camera_height'] = self.agent._env._sim.get_agent_data(0).articulated_agent._sim._sensors["agent_0_articulated_agent_arm_depth"]._sensor_object.node.transformation.translation[1]

	def load_sif_obj_params(self):  
		for k,v in self.agent._env._sim.get_agent_data(0).articulated_agent._sim.ep_info.sif_params['objects_reloc_start_loc'].items():
			self.agent.rom.get_object_by_handle(k).transformation  = mn.Matrix4.translation(v)
			self.agent.rom.get_object_by_handle(k).angular_velocity = mn.Vector3.zero_init()
			self.agent.rom.get_object_by_handle(k).linear_velocity = mn.Vector3.zero_init() 

		self.agent.habitat_env.sim.internal_step(-1)

		targ_handles_sif = self.agent._env._sim.get_agent_data(0).articulated_agent._sim.ep_info.sif_params['objects_reloc_start_loc']
		self.agent.task_phase_room_obj_hard = {targ_handle: self.agent.retr.obj_in_which_room(self.agent.rom.get_object_by_handle(targ_handle).translation) for targ_handle in targ_handles_sif if self.agent.perception.handles_to_catergories[targ_handle.split('_:')[0]] == self.agent.goal_cat}
		self.agent.perception.all_the_handles = self.agent._env._sim.get_agent_data(0).articulated_agent._sim.ep_info.name_to_receptacle.keys()
		self.agent.perception.obj_pose_beginning_of_task_phase = {handle: self.agent.rom.get_object_by_handle(handle).translation for handle in self.agent.perception.all_the_handles} #targ_handle in targ_handles} 


	#For running oracle 
	def get_goal_cat_and_recep(self):
		self.agent.goal_cat = self.agent._env._sim.get_agent_data(0).articulated_agent._sim.ep_info.sif_params['prompt']['initial']['obj_eval']
		initial_prompt = self.agent._env._sim.get_agent_data(0).articulated_agent._sim.ep_info.sif_params['prompt']['initial']


		if self.agent.ep_info.sif_params['prompt']['recep_type'] == 'human':
			task_type = 'human'
		else:
			if initial_prompt['recep_eval_unique']!=None or initial_prompt['recep_eval_notunique']!=None:
				task_type = 'recep'
			else:
				task_type = 'room'
		self.agent.task_type = task_type

		if task_type  == 'human':
			self.agent.goal_recep_or_room = 'human'

		elif task_type =='recep':
			if initial_prompt['recep_eval_unique']!=None:
				self.agent.goal_recep_or_room = initial_prompt['recep_eval_unique']
			else:
				self.agent.goal_recep_or_room = initial_prompt['recep_eval_notunique']
		elif task_type =='room':
			self.agent.goal_recep_or_room =self.agent._env._sim.get_agent_data(0).articulated_agent._sim.ep_info.sif_params['prompt']['initial']['room_function']
		else:
			self.agent.goal_recep_or_room = 'Invalid'

		return self.agent.goal_cat, self.agent.goal_recep_or_room, self.agent.task_type
