import os
import numpy as np
import cv2

from envs.utils.fmm_planner import FMMPlanner
from envs.habitat.eif_env import EIF_Env
import pickle
import torch

from utils.llm_policy_function import get_map_lo_so_dict
from utils.get_human_walk_room_function import get_human_walk_towards_rooms

from agents.submodules import IntermediateStopTracker, LLPlanner, PostProcess, Util, Eval
from constants import categories_to_include


class SIF_Agent(EIF_Env):
	"""The SIF_Agent environment agent class. A seperate SIF_Exp_Env_Agent class
	object is used for each environment thread.

	"""

	def __init__(self, args, rank, config, dataset):

		self.args = args
		config_env = config
		super().__init__(args, rank, config_env, dataset) 
	
		self.planner = Planner(self._env)
		self.intermediate_stop_tracker = IntermediateStopTracker(self)
		self.post_proc = PostProcess(self)
		self.util = Util(self)
		self.eval = Eval(self)
		

	def reset(self, *args, return_info, **kwargs):
		self.viz_goal = None
		self.perception.viz.visited_vis =  np.zeros((720, 720))
		obs, info = super().reset(return_info=return_info, **kwargs)

		self.planner.reset(self.loc, self.args, self.info['room_dict'])

		self.done_fbe = False
		return obs , info



	#Main function
	def plan_act_and_preprocess(self, planner_inputs):
		"""Function responsible for planning, taking the action and
		preprocessing observations

		Args:
			planner_inputs (dict):
				dict with following keys:
					'map_pred'  (ndarray): (M, M) map prediction
					'goal'      (ndarray): (M, M) mat denoting goal locations
					'pose_pred' (ndarray): (7,) array denoting pose (x,y,o)
								 and planning window (gx1, gx2, gy1, gy2)
					 'found_goal' (bool): whether the goal object is found

		Returns:
			obs (ndarray): preprocessed observations ((4+C) x H x W)
			reward (float): amount of reward returned after previous action
			done (bool): whether the episode has ended
			info (dict): contains timestep, pose, goal category and
						 evaluation metric info
		"""
	
		#Initialize 
		self.initialize_tool_status()
		planner_inputs = self.get_human_visible(planner_inputs)
		self.reset_view_angle_and_channel_remove()
		self.util.print_log_text(planner_inputs)
		
		#Plan
		action, task_stop = self.planner.plan_action(planner_inputs, self.args,self.info['ignore_cur_ep'], self.human_walk_dest,self.info['intermediate_stop_human_follow'], self.info['room_dict'], self.timestep, self.fbe_repl_visited_pos_list)
		self.visualize_if_required(planner_inputs)


		#early stop if follow human unnecessarily
		if self.args.task_phase:
			if planner_inputs['cur_tools']['hl_tool'] == 'follow_human' and self.timestep >=50 and self._env._sim.get_agent_data(0).articulated_agent._sim.ep_info.sif_params['ambiguous'] == False:
				breakpoint()
				task_stop = True

		if (self.info['ignore_cur_ep']) and self.timestep >=1:
			task_stop = True
		if not(task_stop):
			# act
			self.reset_intermediate_stops()
			obs, rew, done, self.info = super().step_eif(action)
			obs = self.perception.preprocess_obs(obs, self.sem_seg)  

			#	Postproces and update
			if self.args.task_phase:
				self.update_map_after_step(action, planner_inputs)
				self.intermediate_stop_tracker.check_intermediate_stop(planner_inputs)

			if self.args.premapping_fbe_mode:
				done = self.get_fbe_done()

			self.log_human_agent_pose(planner_inputs)

			
			if self.info['intermediate_stop_human_follow']:
				towards_rooms, plane_array_viz = self.get_human_walk(planner_inputs)
				self.visualize_human_walk(plane_array_viz)
				self.add_human_walk_to_prompt(towards_rooms)

				
			self.obs_shape = obs.shape
			return obs, rew, done, self.info

		else:
			self.task_stop_output()
			done = self.get_done_by_phase()
			if done:
				if not(self.info['ignore_cur_ep']):
					self.util._write_log()
					model_output = self.eval.save_model_task_run_output(write=True)
					already_cond_met = self.eval.evaluate_model(model_output, write=True)

			else:
				self.info['fbe_map_lo_so_dict'] = self.util.get_map_lo_so_dict(planner_inputs['input_dict'], include_human=False)

			self.switch_phase()
			return np.zeros(self.obs_shape), 0., done, self.info

	###################################
	#Initialize and reset before step
	def initialize_tool_status(self):
		self.grab_tool_called = False
		self.put_tool_called = False


	def get_human_visible(self,planner_inputs):
		planner_inputs['human_visible'] = self.retr.human_visible(self.sem_seg)
		return planner_inputs

	def reset_view_angle_and_channel_remove(self):
		self.info['view_angle'] =  0.0 #rot.as_euler('xyz')[0]
		self.info['channel_to_remove'] = None 


	def reset_intermediate_stops(self):
		self.info['intermediate_stop'] = False
		self.info['intermediate_stop_human_follow'] = False

	def _reset_agent_goal(self):
		self._env._sim.get_agent_data(0).articulated_agent.at_goal_ona = False


	###################################
	#Update and finalize after step

	def get_done_by_phase(self):
		if self.args.replay_fbe_actions_phase:
			done = False
		elif self.args.task_phase:
			done = True
		return done
	
	def update_map_after_step(self, action, planner_inputs):
		if action != None:
			if 'hl_tool' in action and action['hl_tool'] in ['grab_obj', 'put_obj']:
				# Determine whether the action is a 'grab' or 'put' based on 'hl_tool'
				is_putting = action['hl_tool'] == 'put_obj'
				# Fetch the object name from planner inputs
				entity_name = planner_inputs['cur_tools']['ll_argument']['obj']
				# Update the map with the appropriate action and entity
				self.post_proc.map_update_put_grab(planner_inputs, action, put=is_putting, entity_name=entity_name)

	def get_fbe_done(self):
		done = self.done_fbe
		if self.done_fbe:
			self.print_log("Done FBE!")
		return done


	def switch_phase(self):
		if self.args.replay_fbe_actions_phase:
			self.args.replay_fbe_actions_phase = False
			self.args.task_phase = True
			self.info['task_phase'] = self.args.task_phase
			self.info['reset_phase2'] = True

		elif self.args.task_phase:
			self.args.replay_fbe_actions_phase = True
			self.args.task_phase = False
			self.info['task_phase'] = self.args.task_phase
	

	def task_stop_output(self):
		self.info["sensor_pose"] = [0., 0., 0.]
		self.info['intermediate_stop'] = True #False



	############################################
	#Log and visualize 
	def visualize_if_required(self, planner_inputs):
		#Draw visited
		if not self.args.replay_fbe_actions_phase:
			self.perception.viz._draw_visited(self.args, self.planner.ll_planner.last_loc, self.planner.start, planner_inputs['map_pred'], self.timestep)
		if self.args.visualize or self.args.print_images:
			self.perception.viz._visualize(planner_inputs, self.planner.start_x, self.planner.start_y, self.viz_goal, self.args, self.episode_id, self.timestep)


	


	def log_human_agent_pose(self, planner_inputs):
		self.info["map_pose_human"] = self.retr.get_map_pose_human()
		self.info["map_pose_agent"] = self.retr.get_map_pose() 
		if self.info['intermediate_stop']:
			self.info['human_pose_room_cur'] = self.retr.get_human_room(planner_inputs)

		#Print human and agent top down distance
		if self.args.task_phase:
			cur_agent_pose = self._env._sim.get_agent_data(0).articulated_agent.sim_obj.translation 
			human_agent_pose = self._env._sim.get_agent_data(1).articulated_agent.sim_obj.translation
			following = planner_inputs['cur_tools']['hl_tool']=='follow_human'
			self.print_log("Following: ",following )
			self.human_agent_distance[self.timestep] = {'following': following , 'dist': self.util.get_top_down_distance(cur_agent_pose, human_agent_pose )}
			self.print_log("HUMAN AGNET DISTANCE: ",self.human_agent_distance[self.timestep])
			self.agent_which_room[self.timestep] = self.retr.get_agent_room(cur_agent_pose)


	###################################
	#Human Tasks
	def get_human_walk(self, planner_inputs):
		self.info['human_pose_room_cur'] = self.retr.get_human_room(planner_inputs)
		towards_rooms, plane_array_viz = get_human_walk_towards_rooms(np.rint(planner_inputs['map_pred']), planner_inputs['sem_map_pred_channels'], self.info['room_dict'], self.args.human_trajectory_index, self.args.threshold_human_walk)
		self.info['human_towards_rooms'] = towards_rooms
		return towards_rooms, plane_array_viz 

	def visualize_human_walk(self, plane_array_viz):
		dump_dir = "{}/dump/{}/".format(self.args.dump_location,self.args.exp_name)
		fn = '{}/episodes/thread_{}/eps_{}/human_walk_viz/{}.png'.format(dump_dir, self.rank, self.episode_id, self.timestep)
		if not os.path.exists(os.path.dirname(fn)):
			os.makedirs(os.path.dirname(fn))
		cv2.imwrite(fn, (plane_array_viz * 255).astype(np.uint8)) 


	def add_human_walk_to_prompt(self, human_towards_rooms):
		prompt = "You saw the human in room " +  str(self.info['human_pose_room_initial']) + ". When you last saw the human, the human was in room " + str(self.info['human_pose_room_cur'])
		if len(human_towards_rooms) >0:
			prompt += ", walking towards rooms " + ", ".join([str(h) for h in self.info['human_towards_rooms']]) + ". \n"
		else:
			prompt += ". \n"
		self.print_log("==============================================\n prompt: \n") 
		self.print_log(prompt)
				



	

	

			
####################################################
#Phase-level planning
class Planner:

	def __init__(self, agent_env):
		self.ll_planner = LLPlanner()
		self.agent_env = agent_env

	def reset(self, loc, args, room_dict):
		self.ll_planner.reset(loc, args, room_dict)




	def plan_action(self, planner_inputs, args, ignore_cur_ep,human_walk_dest,  intermediate_stop_human_follow,  room_dict=None, timestep =None, fbe_repl_visited_pos_list=None):
		if args.premapping_fbe_mode:
			return self.plan_fbe(planner_inputs)
		elif args.task_phase:
			return self.plan_taskphase(timestep, room_dict, planner_inputs, ignore_cur_ep, human_walk_dest, intermediate_stop_human_follow)
		elif args.replay_fbe_actions_phase:
			if timestep <= len(fbe_repl_visited_pos_list)-1:
				self._minimum_plan(planner_inputs)
				action = fbe_repl_visited_pos_list[timestep] #Null action really; doesn't matter
				task_stop = False
			else:
				action=None
				task_stop = True
			return action, task_stop

	def plan_fbe(self, planner_inputs):
		self._minimum_plan(planner_inputs)


		task_stop = False
		
		goal =cur_tool_dict['fbe_goal'] 
		self.viz_goal = goal

		self.done_fbe = self._reset_send_loc(planner_inputs, start, map_pred, goal)

		self.send_loc = np.array([-self.ll_planner.send_loc_goal[0][1], np.array(self.agent_env._sim.get_agent_data(0).articulated_agent.base_pos)[1] , -self.ll_planner.send_loc_goal[0][0]])
		
		action = self.wrap_send_to_loc(self.send_loc) 

		return action,  task_stop

	def plan_taskphase(self, timestep, room_dict, planner_inputs, ignore_cur_ep, human_walk_dest,intermediate_stop_human_follow):
		"""Function responsible for planning

		Args:
			planner_inputs (dict):
				dict with following keys:
					'map_pred'  (ndarray): (M, M) map prediction
					'goal'      (ndarray): (M, M) goal locations
					'pose_pred' (ndarray): (7,) array  denoting pose (x,y,o)
								 and planning window (gx1, gx2, gy1, gy2)
					'found_goal' (bool): whether the goal object is found

		Returns:
			action (int): action id
		"""
		self.start_taskphase = self._minimum_plan(planner_inputs)
		cur_tool_dict = planner_inputs['cur_tools']

		if self._should_end_task(cur_tool_dict, ignore_cur_ep):
			return None, True 

		#These are for prompter
		if cur_tool_dict['hl_tool'] in ["nav_directly_to_goal_pick", "nav_directly_to_goal_put"]:
			action, task_stop = self._process_prompter_action_goal(cur_tool_dict, planner_inputs, human_walk_dest, intermediate_stop_human_follow)

		elif cur_tool_dict['hl_tool'] in ['grab_obj', 'put_obj', "give_human"]:
			action, task_stop = self._process_manipulation_action_goal(cur_tool_dict, planner_inputs, human_walk_dest,intermediate_stop_human_follow)


		elif cur_tool_dict['hl_tool'] in ["nav_to_room", "explore_room", "follow_human"]:
			action, task_stop = self._process_nav_action_goal(timestep, room_dict, cur_tool_dict, planner_inputs,human_walk_dest,intermediate_stop_human_follow)

		elif cur_tool_dict['hl_tool'] == 'no_op':
			action, task_stop = None, False

		return action, task_stop

	def _minimum_plan(self, planner_inputs):
		self.ll_planner.last_loc = self.ll_planner.curr_loc

		# Get Map prediction
		map_pred = np.rint(planner_inputs['map_pred'])
		cur_tool_dict = planner_inputs['cur_tools']

		start_x, start_y, start_o, _, _, _, _ = \
			planner_inputs['pose_pred']

		self.start, self.start_x, self.start_y = self.ll_planner._redefine_currloc_start_startxy(map_pred, start_x, start_y, start_o)
		return self.start

	
		

	def _decide_human_follow_stopped(self): #, planner_inputs):
		human_follow_stopped = self.agent_env._sim.get_agent_data(1).articulated_agent.at_goal_ona
		return human_follow_stopped


	def get_null_action_goal_stop(self, goal):
		action = None
		task_stop = True
		goal[360,360] = 1
		return action, task_stop, goal

	def get_tool_action(self, planner_inputs, tool_type, get_goal_function, wrap_send_to_loc_function, intermediate_stop_human_follow, human_walk_dest, extra_params=None):
		"""
		Handles setting the goal and determining the action based on the tool type.

		Args:
			tool_type (str): The type of tool (e.g., 'follow_human', 'nav_to_room', 'explore_room').
			get_goal_function (method): The method to call to get the goal based on the tool type.
			wrap_send_to_loc_function (method): The method to call to determine the action based on the tool type.
		"""
		# Set the goal using the provided goal retrieval function
		if extra_params is None:
			extra_params = {}

		goal = get_goal_function(**extra_params)
		task_stop = False
		if np.sum(goal)==0:
			action, task_stop, goal = self.get_null_action_goal_stop(goal)

		self.set_and_log_goal(planner_inputs, goal, intermediate_stop_human_follow)
		# Determine the action using the provided send location function
		action = wrap_send_to_loc_function(self.send_loc, human_walk_dest)
		return action, task_stop

	def set_and_log_goal(self, planner_inputs, goal, intermediate_stop_human_follow):
		"""
		Sets the visualized goal and logs send location.

		Args:
			planner_inputs (dict): Planner inputs necessary for resetting send location.
			goal (ndarray): The computed goal to be visualized and used for setting send location.
		"""
		self.viz_goal = goal
		self.ll_planner._reset_send_loc_for_obj_or_point_goal(planner_inputs, self.start, planner_inputs['map_pred'], goal, intermediate_stop_human_follow,put_grab_navloc=True)
		self.send_loc = np.array([
			-self.ll_planner.send_loc_goal[0][1],
			np.array(self.agent_env._sim.get_agent_data(0).articulated_agent.base_pos)[1],
			-self.ll_planner.send_loc_goal[0][0]
		])
		#self.print_log("send loc is ", self.send_loc)

	def _process_manipulation_action_goal(self, cur_tool_dict, planner_inputs, human_walk_dest, intermediate_stop_human_follow):
		"""
		Processes actions for 'grab_obj', 'put_obj', and 'give_human' based on tool conditions.

		Args:
			cur_tool_dict (dict): Current tool dictionary containing tool and execution details.
			planner_inputs (dict): Dictionary containing all necessary inputs for planning actions.
		"""
		if cur_tool_dict['whether_execute']:
			self.agent_env._sim.get_agent_data(0).articulated_agent.at_goal_ona = False

		tool_type = cur_tool_dict['hl_tool']
		if tool_type in ['grab_obj', 'put_obj']:
			return self.process_obj_goal_actions(tool_type, cur_tool_dict, planner_inputs, human_walk_dest, intermediate_stop_human_follow)
		elif tool_type == "give_human":
			return self.process_give_human_action(cur_tool_dict)

	def process_obj_goal_actions(self, tool_type, cur_tool_dict, planner_inputs,human_walk_dest, intermediate_stop_human_follow):
		"""
		Handles setting and processing goals for 'grab_obj' and 'put_obj'.

		Args:
			tool_type (str): Specific tool type ('grab_obj' or 'put_obj').
			cur_tool_dict (dict): Current tool dictionary containing tool and execution details.
			planner_inputs (dict): Dictionary containing all necessary inputs for planning actions.

		Returns:
			action: The computed action based on tool type and visibility.
		"""
		objgoal_params = {
			'put_obj': {'room_dilation': 10, 'goal_dilation': 0},
			'grab_obj': {'room_dilation': 5, 'goal_dilation': 3}
		}

		task_stop = False
		if not self.agent_env._sim.get_agent_data(0).articulated_agent.at_goal_ona:
			if cur_tool_dict['whether_execute']:
				goal_settings = objgoal_params[tool_type]
				goal_settings['categories_to_include'] = categories_to_include
				goal_settings['cur_tool_dict'] = cur_tool_dict
				goal_settings['planner_inputs'] = planner_inputs
				goal = self.ll_planner.get_obj_goal(**goal_settings)

				if np.sum(goal) == 0:
					action, task_stop, goal = self.get_null_action_goal_stop(goal)

				self.set_and_log_goal(planner_inputs, goal, intermediate_stop_human_follow)
			action = self.ll_planner.wrap_just_rotate_cam_down(self.send_loc, human_walk_dest)
		else:
			action = {'hl_tool': tool_type, 'args': cur_tool_dict['ll_argument']['obj']}  
		return action, task_stop


	def process_give_human_action(self, cur_tool_dict):
		"""
		Handles the 'give_human' action by returning a predefined action structure.

		Args:
			cur_tool_dict (dict): Current tool dictionary containing the 'give_human' action details.

		Returns:
			action: The predefined action for 'give_human'.
		"""
		task_stop = False
		return {'hl_tool': "give_human", 'args': 'human'}, task_stop



	

	def _should_end_task(self, cur_tool_dict, ignore_cur_ep):
		"""
		Determines whether the task should end based on the current tool or episode info.

		Returns:
			bool: True if the task should end, False otherwise.
		"""
		return cur_tool_dict['hl_tool'] == "END_TASK" or ignore_cur_ep 




	def _process_prompter_action_goal(self,  cur_tool_dict, planner_inputs,human_walk_dest, intermediate_stop_human_follow):
		"""
		Process navigation to goals for actions like picking or putting objects based on whether the agent is at the goal or not.

		Args:
			cur_tool_dict (dict): Current tool information.
			planner_inputs (dict): Inputs necessary for planning actions.

		Returns:
			tuple: (action, task_stop flag)
		"""
		objgoal_params = {
						'nav_directly_to_goal_put': {'room_dilation': 0, 'goal_dilation': 0},
						'nav_directly_to_goal_pick': { 'room_dilation': 0, 'goal_dilation': 3}
					}

		task_stop = False
		tool_type = cur_tool_dict['hl_tool']
		if not self.agent_env._sim.get_agent_data(0).articulated_agent.at_goal_ona:
			if cur_tool_dict['whether_execute']:
				goal_settings = objgoal_params[tool_type]
				goal_settings['categories_to_include'] = categories_to_include
				goal_settings['cur_tool_dict'] = cur_tool_dict
				goal_settings['planner_inputs'] = planner_inputs
				goal = self.ll_planner.get_obj_goal(**goal_settings)

				if np.sum(goal) == 0:
					action, task_stop, goal = self.get_null_action_goal_stop(goal)

				self.set_and_log_goal(planner_inputs, goal, intermediate_stop_human_follow)

				#WTF is this?
				#goal = self.__goal(cur_tool_dict, planner_inputs) 
			action = self.ll_planner.wrap_just_rotate_cam_down(self.send_loc, human_walk_dest)

		else:
			action = {'hl_tool': cur_tool_dict['hl_tool'].replace('nav_directly_to_goal_', '').replace('pick', 'grab') + '_obj', 'args': cur_tool_dict['ll_argument']['obj']}
		return action, task_stop


	def _process_nav_action_goal(self, timestep, room_dict, cur_tool_dict, planner_inputs,human_walk_dest, intermediate_stop_human_follow):
		tool_actions = {
			"follow_human": (self.ll_planner.get_follow_human_goal, self.ll_planner.wrap_send_to_loc_follow_human),
			"nav_to_room": (self.ll_planner.get_nav_goal, self.ll_planner.wrap_send_to_loc),
			"explore_room": (self.ll_planner.get_explore_goal, self.ll_planner.wrap_send_to_loc_search)
		}

		if cur_tool_dict['hl_tool'] == "follow_human":
			extra_params = {'intermediate_stop_human_follow': intermediate_stop_human_follow,
							'room_dict': room_dict, 'planner_inputs': planner_inputs, 'human_visible': planner_inputs['human_visible'], 'start': self.start}
		elif cur_tool_dict['hl_tool'] == "nav_to_room":
			extra_params = {'cur_tool_dict': cur_tool_dict, 'map_pred': planner_inputs['map_pred']}
		elif cur_tool_dict['hl_tool'] == "explore_room":
			extra_params = {'timestep': timestep, 'cur_tool_dict': cur_tool_dict, 'map_pred': planner_inputs['map_pred'], 'room_dict': room_dict}

		get_goal_function, wrap_send_to_loc_function = tool_actions[cur_tool_dict['hl_tool']]
		action, task_stop = self.get_tool_action(planner_inputs, cur_tool_dict['hl_tool'], get_goal_function, wrap_send_to_loc_function, intermediate_stop_human_follow, human_walk_dest, extra_params)
		return action, task_stop

		


			