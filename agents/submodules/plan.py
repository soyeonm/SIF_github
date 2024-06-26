import os

import envs.utils.pose as pu
from constants import local_w, step_action
from envs.utils.fmm_planner import FMMPlanner

from skimage.measure import regionprops
import skimage.morphology

import numpy as np
import torch

import copy

from constants import human_trajectory_index

from utils.get_human_walk_room_function import get_ahead_point


class LLPlanner:
	def __init__(self):
		self.selem = skimage.morphology.disk(4) #5)
		self.visited = None
		self.curr_loc = None
		self.last_loc = None


	def reset(self, loc, args, room_dict):
		# Episode initializations
		map_shape = (args.map_size_cm // args.map_resolution,
					 args.map_size_cm // args.map_resolution)
		self.visited = np.zeros(map_shape)
		self.curr_loc = [args.map_size_cm / 100.0 / 2.0,
						 args.map_size_cm / 100.0 / 2.0, 0.]

		self.stg = None
		self.prev_fmm = None		
		self.loc = loc 
		self.args = args
		self.room_dict = room_dict
		self.fbe_goal_mat = None




	#0. Reset, Redefine
	#Redfine
	def _redefine_currloc_start_startxy(self, map_pred, start_x, start_y, start_o):
		args = self.args

		# Get curr loc
		self.curr_loc = [start_x, start_y, start_o]
		r, c = start_y, start_x
		start = [int(r * 100.0 / args.map_resolution),
				 int(c * 100.0 / args.map_resolution)]
		start = pu.threshold_poses(start, map_pred.shape)
		
		rel_pose_change = np.array(pu.get_rel_pose_change(self.loc.get_sim_agent_location(), self.loc.agent_0_starting_camera_pose_in_agent_coordinate))*100/5
		rpc_ori = np.array(pu.get_rel_pose_change(self.loc.get_sim_agent_location(), self.loc.agent_0_starting_camera_pose_in_agent_coordinate))
		guessed_sim_agent_location = self.loc.get_new_pose_batch(torch.tensor(self.loc.agent_0_starting_camera_pose_in_agent_coordinate).unsqueeze(0), torch.tensor(rpc_ori).unsqueeze(0))


		rel_pose_change = [rel_pose_change[0], rel_pose_change[1], rel_pose_change[2]]
		new_start_1, new_start_0, _ = (np.array(rel_pose_change) + np.array([local_w/2, local_w/2, 0])).astype(int)
		start = [new_start_0, new_start_1]
		rp = np.array(pu.get_rel_pose_change(self.loc.get_sim_agent_location(), self.loc.agent_0_starting_camera_pose_in_agent_coordinate))
		ori_start_x, ori_start_y, _ = (rp + np.array([local_w/2, local_w/2, 0])/ 20.0)

		self.curr_loc = [ori_start_x, ori_start_y, start_o]
		start_x, start_y = ori_start_x, ori_start_y

		self.visited[start[0] - 0:start[0] + 1,
									   start[1] - 0:start[1] + 1] = 1


		return start, start_x, start_y


	#Send to low level
	def _reset_send_loc_for_obj_or_point_goal(self, planner_inputs, start, map_pred, goal, intermediate_stop_human_follow, put_grab_navloc=False):
		run_condition_task = self.args.task_phase and ((planner_inputs['cur_tools']['whether_execute']) or intermediate_stop_human_follow)
		
		#Now set it 
		if run_condition_task:
			np.where(goal)
			min_goal = (np.where(goal)[0][0], np.where(goal)[1][0]) 
			self.stg = min_goal

			rpc_min_goal = np.array(pu.get_rel_pose_change((min_goal[0], min_goal[1], 0.0), (local_w/2, local_w/2, 0.0))) 
			rpc_min_goal = np.array([rpc_min_goal[1] * 0.05, rpc_min_goal[0] * 0.05, rpc_min_goal[2]])
			guessed_min_goal_sim_location = self.loc.get_new_pose_batch(torch.tensor(self.loc.agent_0_starting_camera_pose_in_agent_coordinate).unsqueeze(0), torch.tensor(rpc_min_goal).unsqueeze(0))

			self.send_loc_goal = guessed_min_goal_sim_location 



	#For fbe 
	def _reset_send_loc(self, timestep, planner_inputs, start, map_pred, goal, intermediate_stop_human_follow, put_grab_navloc=False):
		stop = False
		if not(self.stg == None):
			stop_cond = np.linalg.norm((np.array([start[0], start[1]]) - self.stg))<= self.config.task.actions['agent_0_oracle_nav_with_backing_up_action']['dist_thresh']*100/5

		prev_goal_broken = False
		if not(self.stg == None):
			prev_goal_broken = self._check_if_goal_broken(map_pred, np.rint(planner_inputs['exp_pred']), start, np.copy(goal))


		done_fbe = False
		run_condition_premapping_fbe_mode = self.args.premapping_fbe_mode and (timestep ==0 or timestep % self.args.num_local_steps==1 or prev_goal_broken)
		run_condition_replay_fbe_actions_phase = self.args.replay_fbe_actions_phase and timestep ==0
		run_condition_task = self.args.task_phase and (planner_inputs['cur_tools']['whether_execute'] or prev_goal_broken  or intermediate_stop_human_follow )#
		run_conidition_scripted_action_phase = self.args.replay_scripted_actions_phase and timestep ==0
		if run_condition_premapping_fbe_mode or run_condition_replay_fbe_actions_phase or run_condition_task or run_conidition_scripted_action_phase:
			stg, min_goal, stop = self._get_stg(map_pred, np.rint(planner_inputs['exp_pred']), start, np.copy(goal),planning_window, put_grab_navloc=put_grab_navloc)
			if (self._planner_broken(self.planner_detect_broken.fmm_dist, start) and self.args.premapping_fbe_mode):# or self.failed_spot_init:
				done_fbe = True

			self.stg = min_goal

			#call just to get self.prev_fmm
			goal_broken = self._check_if_goal_broken(map_pred, np.rint(planner_inputs['exp_pred']), start, np.copy(goal)) #This comes out as true

			(stg_x, stg_y) = stg 
			rpc_stg = np.array(pu.get_rel_pose_change((stg[0], stg[1], 0.0), (local_w/2, local_w/2, 0.0))) 
			rpc_stg = np.array([rpc_stg[1] * 0.05, rpc_stg[0] * 0.05, rpc_stg[2]])
			guessed_stg_sim_agent_location = self.loc.get_new_pose_batch(torch.tensor(self.loc.agent_0_starting_camera_pose_in_agent_coordinate).unsqueeze(0), torch.tensor(rpc_stg).unsqueeze(0))

			rpc_min_goal = np.array(pu.get_rel_pose_change((min_goal[0], min_goal[1], 0.0), (local_w/2, local_w/2, 0.0))) 
			rpc_min_goal = np.array([rpc_min_goal[1] * 0.05, rpc_min_goal[0] * 0.05, rpc_min_goal[2]])
			guessed_min_goal_sim_location = self.loc.get_new_pose_batch(torch.tensor(self.loc.agent_0_starting_camera_pose_in_agent_coordinate).unsqueeze(0), torch.tensor(rpc_min_goal).unsqueeze(0))

			self.send_loc_goal = guessed_min_goal_sim_location 
		return done_fbe

	#1. For each tool, get goals in map 
	def get_point_goal(self, goal):
		connected_regions = skimage.morphology.label(goal, connectivity=2)
		for c in np.unique(connected_regions):
			if c>0 and np.sum((connected_regions == c) * goal)>0:  
				get_centroid_from = (((connected_regions == c) * goal )>0).astype(np.uint8)
				properties = regionprops(get_centroid_from, get_centroid_from)
				center_of_mass = properties[0].centroid
				goal = np.zeros(goal.shape)
				goal[int(center_of_mass[0]), int(center_of_mass[1])] = 1
		return goal


	def get_obj_goal(self, categories_to_include, cur_tool_dict, planner_inputs, room_dilation=0, goal_dilation=0):
		obj_channel_entry = categories_to_include[cur_tool_dict['ll_argument']['obj']] 
		goal = 1.0 * planner_inputs['sem_map_pred_channels'][obj_channel_entry].cpu().numpy()

		if goal_dilation>0:
			so_selem = skimage.morphology.disk(goal_dilation) #3)
			goal = 1.0 * skimage.morphology.binary_dilation(goal , so_selem)

		if room_dilation>0:
			wheres_room_ori = cur_tool_dict['ll_argument']['in_room']
			selem = skimage.morphology.disk(room_dilation) #5)
			wheres_room = copy.deepcopy(wheres_room_ori)
			wheres_room = skimage.morphology.binary_dilation(wheres_room , selem)
			goal =  1.0 * (goal * wheres_room)

		goal = self.get_point_goal(goal)
		return goal
	
	def get_nav_goal(self, cur_tool_dict, map_pred):
		goal = cur_tool_dict['ll_argument']
		if cur_tool_dict['whether_execute']:
			traversible = skimage.morphology.binary_dilation(map_pred, self.selem) != True					
			traversible[self.visited == 1] = 1

			#Get the center of mass among traversible 
			if np.sum((cur_tool_dict['ll_argument'] * traversible))>0:
				get_centroid_from = ((cur_tool_dict['ll_argument'] * traversible) >0).astype(np.uint8)
				properties = regionprops(get_centroid_from, get_centroid_from)
				center_of_mass = properties[0].centroid
				goal = np.zeros(goal.shape)
				goal[int(center_of_mass[0]), int(center_of_mass[1])] = 1

				goal = goal * 1.0
			else:
				goal = goal * 1.0
			self.prev_goal = goal
		else:
			goal = self.prev_goal
		return goal

	def get_explore_goal(self, timestep, cur_tool_dict,room_dict, map_pred):
		if cur_tool_dict['whether_execute']:
			chosen_room_number = cur_tool_dict['chosen_room_number']
			#Then choose a random where from the goal
			np.random.seed(timestep)
			wheres = np.where(room_dict['room_assignment'][chosen_room_number])
			len_where = len(wheres[0])
			#np.random.seed(seed)
			where_idx = np.random.choice(np.arange(len_where))
			global_goal = [wheres[0][where_idx], wheres[1][where_idx]]
			goal = np.zeros(map_pred.shape)
			goal[global_goal[0], global_goal[1]] = 1
			self.search_turn_count = 0
			self.prev_goal = goal
		else:
			goal = self.prev_goal #self.explore_goal
			self.search_turn_count +=1
		return goal

	def get_follow_human_goal(self, planner_inputs, intermediate_stop_human_follow, room_dict, human_visible, start):
		map_pred = planner_inputs['map_pred']
		if planner_inputs['sem_map_pred_channels'][human_trajectory_index, :, :].max().item() !=0:
			new_end_point = None

			if intermediate_stop_human_follow and  not(human_visible):
				new_end_point = get_ahead_point(np.rint(planner_inputs['map_pred']), planner_inputs['sem_map_pred_channels'], room_dict, human_trajectory_index)
				if not(new_end_point is None) and not(np.isnan(new_end_point[0]) or np.isnan(new_end_point[1])):
					goal = np.zeros(map_pred.shape)
					goal[int(new_end_point[0]), int(new_end_point[1])] = 1

			if (new_end_point is None) or (np.isnan(new_end_point[0]) or np.isnan(new_end_point[1]))\
			 or not(intermediate_stop_human_follow and  not(human_visible)):
				#Then put where the human last was as the goal
				numpy_traj = planner_inputs['sem_map_pred_channels'][human_trajectory_index, :, :].cpu().numpy()
				wheres_person_last = np.where(numpy_traj == planner_inputs['sem_map_pred_channels'][human_trajectory_index, :, :].max().item())
				goal = np.zeros(map_pred.shape)
				goal[wheres_person_last[0], wheres_person_last[1]] = 1 
		else:
			#Just set the goal as the current pose 
			goal = np.zeros(map_pred.shape)
			goal[start[0], start[1]] = 1
		return goal


	#2. Planning inside map
	def _planner_broken(self, fmm_dist, start):
		if fmm_dist[start[0], start[1]] == fmm_dist.max():
			return True
		else:
			return False

	def _check_if_goal_broken(self, grid, exp_grid, start, goal):
		
		x1, y1, = 0, 0
		x2, y2 = grid.shape


		exp_pred = exp_grid[x1:x2, y1:y2]
		traversible = skimage.morphology.binary_dilation(
			grid[x1:x2, y1:y2],
			self.selem) != True
		

		traversible[self.visited[gx1:gx2, gy1:gy2][x1:x2, y1:y2] == 1] = 1

		traversible[int(start[0] - x1) - 5:int(start[0] - x1) + 6,
					int(start[1] - y1) - 5:int(start[1] - y1) + 6] = 1


		exp_pred = exp_pred * traversible 

		goal = np.zeros(traversible.shape)
		goal[self.stg[0], self.stg[1]] = 1

		planner = FMMPlanner(traversible, step_size=40)

		selem = skimage.morphology.disk(10)
		goal = skimage.morphology.binary_dilation(
			goal, selem) != True
		goal = 1 - goal * 1.

		planner.set_multi_goal(goal)
		try:
			planner_broken =  planner.fmm_dist[start[0], start[1]] == planner.fmm_dist.max()
		except:
			breakpoint()

		if not(self.prev_fmm == None):
			goal_outisde_slightly = planner.fmm_dist[start[0], start[1]] - self.prev_fmm >= 50
		else:
			goal_outisde_slightly = False
		self.prev_fmm = copy.deepcopy(planner.fmm_dist[start[0], start[1]])
		return planner_broken or goal_outisde_slightly 



	def _get_stg(self, grid, exp_grid, start, goal, put_grab_navloc=False):
		"""Get short-term goal"""
		

		x1, y1, = 0, 0
		x2, y2 = grid.shape

		exp_pred = exp_grid[x1:x2, y1:y2]
		traversible = skimage.morphology.binary_dilation(
			grid[x1:x2, y1:y2],
			self.selem) != True

		traversible[self.visited[gx1:gx2, gy1:gy2][x1:x2, y1:y2] == 1] = 1

		traversible[int(start[0] - x1) - 5:int(start[0] - x1) + 6,
					int(start[1] - y1) - 5:int(start[1] - y1) + 6] = 1


		exp_pred = exp_pred * traversible 

		state = [start[0] - x1, start[1] - y1]
		around_agent = np.zeros(exp_pred.shape)
		around_agent[state[0], state[1]] = 1
		selem = skimage.morphology.disk(30)
		around_agent = skimage.morphology.binary_dilation(around_agent, selem)
		if np.sum(exp_pred * (1. - around_agent))>0:
			exp_pred = exp_pred * (1. - around_agent)

		goal_ori = copy.deepcopy(goal)

		planner = FMMPlanner(traversible, step_size=40)
		self.planner_detect_broken = FMMPlanner(traversible, step_size=5)
		if self.args.premapping_fbe_mode:
			selem = skimage.morphology.disk(60)
			goal = skimage.morphology.binary_erosion(goal, selem) != True         
			goal = 1 - goal * 1.


		else:
			if put_grab_navloc:
				selem = skimage.morphology.disk(10)
				goal = skimage.morphology.binary_dilation(
					goal, selem) != True
				goal = 1 - goal * 1.

			try:
				planner.set_multi_goal(goal)
				planner_broken = self._planner_broken(planner.fmm_dist, start)
				while planner_broken:
					goal = skimage.morphology.binary_dilation(goal,selem) *1.0
					planner.set_multi_goal(goal)
					planner_broken = self._planner_broken(planner.fmm_dist, start)
			except:
				breakpoint()
		


		if self.args.premapping_fbe_mode:
			selem = skimage.morphology.disk(30)
			
			goal_dil = skimage.morphology.binary_dilation(goal, selem) == True
			goal = goal_dil
			self.fbe_goal_mat = goal



		self.planner_detect_broken.set_multi_goal(goal)

		state = [start[0] - x1, start[1] - y1]
		if self.args.premapping_fbe_mode :
			state_mat = np.zeros(goal_dil.shape)
			state_mat[state[0], state[1]] = 1
			planner.set_multi_goal(state_mat)
			mul = goal_dil*planner.fmm_dist
			mul[goal_dil==0] = np.max(mul) + 1
			goal_min_x, goal_min_y = np.unravel_index(np.argmin(mul), mul.shape)
			if self.args.proj_exp:
				goal_minxy_mat = np.zeros(goal_dil.shape)
				goal_minxy_mat[goal_min_x, goal_min_y] = 1
				planner.set_multi_goal(goal_minxy_mat)
				exp_mul = exp_pred * planner.fmm_dist
				exp_mul[exp_pred==0] = np.max(exp_mul) + 1
				goal_min_x, goal_min_y = np.unravel_index(np.argmin(exp_mul), mul.shape)
				#

			planner.set_multi_goal(goal)
			stg_x, stg_y, _, stop = planner.get_short_term_goal(state)
			

		else:
			state_mat = np.zeros(goal.shape)
			state_mat[state[0], state[1]] = 1

			planner.set_multi_goal(state_mat)
			mul = goal*planner.fmm_dist
			mul[goal==0] = np.max(mul) + 1
			goal_min_x, goal_min_y = np.unravel_index(np.argmin(mul), mul.shape)
			stg_x, stg_y = goal_min_x, goal_min_y
			stop = False

		

		stg_x, stg_y = stg_x + x1 , stg_y + y1 
		goal_min_x, goal_min_y = goal_min_x+ x1 , goal_min_y + y1 


		return (stg_x, stg_y), (goal_min_x, goal_min_y), stop


	
	#3. Wrappers
	#Wrap to low level
	def wrap_send_to_loc(self, send_to_habitat, human_walk_dest):
		action_list = step_action
		#Replica
		action_args = {'agent_0_arm_action': np.array([0., 0., 0., 0., 0., 0., 0.]), 'agent_0_grip_action':np.array([0.]), 'agent_0_base_vel': np.array([0., 0.]), 'agent_0_oracle_nav_action': np.array([0.]), 'agent_0_pddl_action': np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
			0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
			0., 0.]), 'agent_0_rearrange_stop': np.array([0.]), 

		'agent_1_base_vel': np.array([0., 0.]), 'agent_1_oracle_nav_with_backing_up_action': np.array([np.inf]), 'agent_1_pddl_action': np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
			0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
			0., 0.]), 'agent_1_rearrange_stop': np.array([0.])}

		
		action_args['agent_0_oracle_nav_with_backing_up_action'] = send_to_habitat 
		if not(human_walk_dest is None):
			action_args['agent_1_oracle_nav_with_backing_up_action'] = human_walk_dest 

		step_dict = {'action': action_list, 'action_args': action_args} 
		return step_dict

	def wrap_send_to_loc_follow_human(self, send_to_habitat, human_walk_dest):
		action_list = step_action
		#Replica
		action_args = {'agent_0_arm_action': np.array([0., 0., 0., 0., 0., 0., 0.]), 'agent_0_grip_action':np.array([0.]), 'agent_0_base_vel': np.array([0., 0.]), 'agent_0_oracle_nav_action': np.array([0.]), 'agent_0_pddl_action': np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
			0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
			0., 0.]), 'agent_0_rearrange_stop': np.array([0.]), 

		'agent_1_base_vel': np.array([0., 0.]), 'agent_1_oracle_nav_with_backing_up_action': np.array([np.inf]), 'agent_1_pddl_action': np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
			0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
			0., 0.]), 'agent_1_rearrange_stop': np.array([0.])}
		
		action_args['agent_0_oracle_nav_with_backing_up_action'] = send_to_habitat 
		if not(human_walk_dest is None):
			action_args['agent_1_oracle_nav_with_backing_up_action'] = human_walk_dest 
		action_args['agent_0_human_follow'] = 1.0 

		step_dict = {'action': action_list, 'action_args': action_args} 
		return step_dict


	def wrap_just_rotate_cam_down(self, goal_obj_habitat,human_walk_dest):
		action_list = step_action
		#Replica
		action_args = {'agent_0_arm_action': np.array([0., 0., 0., 0., 0., 0., 0.]), 'agent_0_grip_action':np.array([0.]), 'agent_0_base_vel': np.array([0., 0.]), 'agent_0_oracle_nav_action': np.array([0.]), 'agent_0_pddl_action': np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
			0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
			0., 0.]), 'agent_0_rearrange_stop': np.array([0.]), 

		'agent_1_base_vel': np.array([0., 0.]), 'agent_1_oracle_nav_with_backing_up_action': np.array([np.inf]), 'agent_1_pddl_action': np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
			0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
			0., 0.]), 'agent_1_rearrange_stop': np.array([0.])}

		
		action_args['agent_0_oracle_nav_with_backing_up_action'] = goal_obj_habitat 
		if not(self.args.magic_grasp_and_put):
			action_args['agent_0_just_rotate'] = 1.0 
		if not(human_walk_dest is None):
			action_args['agent_1_oracle_nav_with_backing_up_action'] =human_walk_dest

		step_dict = {'cam_down': True, 'action': action_list, 'action_args': action_args} 

		return step_dict

	def wrap_send_to_loc_search(self, goal_obj_habitat,human_walk_dest, human_send_to_habitat=None):
		action_list = step_action
		#Replica
		action_args = {'agent_0_arm_action': np.array([0., 0., 0., 0., 0., 0., 0.]), 'agent_0_grip_action':np.array([0.]), 'agent_0_base_vel': np.array([0., 0.]), 'agent_0_oracle_nav_action': np.array([0.]), 'agent_0_pddl_action': np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
			0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
			0., 0.]), 'agent_0_rearrange_stop': np.array([0.]), 

		'agent_1_base_vel': np.array([0., 0.]), 'agent_1_oracle_nav_with_backing_up_action': np.array([np.inf]), 'agent_1_pddl_action': np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
			0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
			0., 0.]), 'agent_1_rearrange_stop': np.array([0.])}

		
		action_args['agent_0_oracle_nav_with_backing_up_action'] = goal_obj_habitat 
		action_args['agent_0_search'] = 1.0 
		if not(human_walk_dest is None):
			action_args['agent_1_oracle_nav_with_backing_up_action'] = human_walk_dest 

		step_dict = {'cam_down': True, 'action': action_list, 'action_args': action_args} 
		return step_dict