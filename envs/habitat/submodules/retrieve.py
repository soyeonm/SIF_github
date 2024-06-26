import skimage.morphology
from skimage.measure import regionprops
from constants import local_w

import envs.utils.pose as pu
from envs.utils.fmm_planner import FMMPlanner


import numpy as np
import torch
import copy
import magnum as mn


class Retrieve:
	def __init__(self, args, env, loc):
		self.loc = loc
		self.args = args
		self.agent_env = env
		self.local_w = local_w

	def reset(self, room_dict):
		self.room_dict = room_dict


	def get_map_pose_any_pose(self, any_pose):
		rel_pose_change = np.array(pu.get_rel_pose_change(any_pose, self.loc.agent_0_starting_camera_pose_in_agent_coordinate))*100/5
		rpc_ori = np.array(pu.get_rel_pose_change(any_pose, self.loc.agent_0_starting_camera_pose_in_agent_coordinate))
		guessed_sim_agent_location = self.loc.get_new_pose_batch(torch.tensor(self.loc.agent_0_starting_camera_pose_in_agent_coordinate).unsqueeze(0), torch.tensor(rpc_ori).unsqueeze(0))

		rel_pose_change = [rel_pose_change[0], rel_pose_change[1], rel_pose_change[2]]
		new_start_1, new_start_0, _ = (np.array(rel_pose_change) + np.array([self.local_w/2, self.local_w/2, 0])).astype(int)
		start = [new_start_0, new_start_1]
		return start


	def get_map_pose_obj_pose(self, any_pose_translation, any_pose_rotation):
		current_pose_base = np.array(any_pose_translation)[[0,2]] #x,y
		current_pose_rot = float(any_pose_rotation.angle())
		current_pose = np.array([current_pose_base[0], current_pose_base[1], current_pose_rot])
		rel_pose_change = np.array(pu.get_rel_pose_change(current_pose, self.loc.agent_0_starting_camera_pose))*100/5
		rel_pose_change = [rel_pose_change[0], -rel_pose_change[1], rel_pose_change[2]]
		full_map_cur_pose_on_map = (np.array(rel_pose_change) + np.array([self.local_w/2 , self.local_w/2, 0])).astype(int)

		return full_map_cur_pose_on_map


	def get_map_pose_human(self):
		rel_pose_change = np.array(pu.get_rel_pose_change(self.loc.get_sim_human_location(), self.loc.agent_0_starting_camera_pose_in_agent_coordinate))*100/5
		rpc_ori = np.array(pu.get_rel_pose_change(self.loc.get_sim_human_location(), self.loc.agent_0_starting_camera_pose_in_agent_coordinate))
		guessed_sim_agent_location = self.loc.get_new_pose_batch(torch.tensor(self.loc.agent_0_starting_camera_pose_in_agent_coordinate).unsqueeze(0), torch.tensor(rpc_ori).unsqueeze(0))

		rel_pose_change = [rel_pose_change[0], rel_pose_change[1], rel_pose_change[2]]
		new_start_1, new_start_0, _ = (np.array(rel_pose_change) + np.array([self.local_w/2, self.local_w/2, 0])).astype(int)
		start = [new_start_0, new_start_1]
		return start


	def get_map_pose(self):
		rel_pose_change = np.array(pu.get_rel_pose_change(self.loc.get_sim_agent_location(), self.loc.agent_0_starting_camera_pose_in_agent_coordinate))*100/5
		rpc_ori = np.array(pu.get_rel_pose_change(self.loc.get_sim_agent_location(), self.loc.agent_0_starting_camera_pose_in_agent_coordinate))
		guessed_sim_agent_location = self.loc.get_new_pose_batch(torch.tensor(self.loc.agent_0_starting_camera_pose_in_agent_coordinate).unsqueeze(0), torch.tensor(rpc_ori).unsqueeze(0))

		rel_pose_change = [rel_pose_change[0], rel_pose_change[1], rel_pose_change[2]]
		new_start_1, new_start_0, _ = (np.array(rel_pose_change) + np.array([self.local_w/2, self.local_w/2, 0])).astype(int)
		start = [new_start_0, new_start_1]
		# start_x, start_y = ori_start_x, ori_start_y
		return start


	#0. Robot agent
	def get_agent_room(self,agent_translation):
		pose = (-agent_translation[2], -agent_translation[0], 0)
		goal_cat_map_pose  = self.get_map_pose_any_pose(pose) 
		room = self.get_room_from_pose(None, goal_cat_map_pose) 
		return room

	def get_room_from_pose(self, full_map, map_pose):
		cur_pose_room = None
		for r, r_map in self.room_dict['room_assignment'].items():
			try:
				if not(r in ['free_space', 'free space']) and r_map[map_pose[0], map_pose[1]]==1:
					cur_pose_room = r
			except:
				breakpoint()
		if cur_pose_room is None:
			planner = FMMPlanner(np.ones((720, 720))) 
			r_name2dist = {}
			for r, r_map in self.room_dict['room_assignment'].items():
				if not(r in ['free_space', 'free space']):
					planner.set_multi_goal(r_map)
					dist = planner.fmm_dist[map_pose[0], map_pose[1]]
					r_name2dist[r] = dist
			#Get the lowest
			room_min = min(r_name2dist, key=lambda k: r_name2dist[k])
			if r_name2dist[room_min] < 0.5 * 100 / 5.:
				cur_pose_room = room_min 
		if cur_pose_room == None:
			cur_pose_room ='free_space'
		return cur_pose_room


	def human_spot_same_room(self): 
		human_translation = self.agent_env._sim.agents_mgr[1].articulated_agent.sim_obj.transformation.translation
		agent_translation = self.agent_env._sim.agents_mgr[0].articulated_agent.sim_obj.transformation.translation
		translations = [agent_translation, human_translation]
		poses = [(-t[2], -t[0], 0) for t in translations]
		goal_cat_map_pose  = [self.get_map_pose_any_pose(pose) for pose in poses]
		rooms = [self.get_room_from_pose(None, pose) for pose in goal_cat_map_pose ]
		return not(rooms[1] in ['free_space', 'free space']) and (rooms[0] == rooms[1])


	def get_human_room(self, planner_inputs):
		human_traj_map = planner_inputs['sem_map_pred_channels'][self.args.human_trajectory_index].cpu().numpy()
		most_recent_step = int(human_traj_map.max())
		get_centroid_from = (human_traj_map==most_recent_step).astype(np.uint8)
		properties = regionprops(get_centroid_from, get_centroid_from)
		end_center_of_mass = properties[0].centroid
		human_room = self.get_room_from_pose(None, [int(end_center_of_mass[0]), int(end_center_of_mass[1])])

		return human_room



	def human_visible(self, sem_seg): 
		self.human_mask = sem_seg[:, :, self.args.human_sem_index]
		self.cur_human_visible = np.sum(self.human_mask) >0
		return np.sum(self.human_mask) >0



	#2. Objects
	def obj_on_recep(self, obj_translation, recep_instance):
		success = False
		recep_up = mn.Vector3.y_axis(1.0)
		trans_offset = 0.08

		for sample_i in range(500):
			sampled_object_position = recep_instance.sample_uniform_global(self.agent_env._sim, 1.0) + recep_up*trans_offset
			dist = np.linalg.norm(sampled_object_position - obj_translation , ord=2) 
			if dist < 0.15:
				success = True
		return success

	def obj_in_which_room(self, obj_translation):
		obj_trans_pose = (-obj_translation[2], -obj_translation[0], 0)
		obj_map_pose = self.get_map_pose_any_pose(obj_trans_pose)
		which_room = self.get_room_from_pose(None, obj_map_pose)
		return which_room


	def obj_in_room(self,  obj_translation, obj_rotation, room_function, grasper_closed):
		function2_roomnum = {}
		for k, v in self.room_dict['human_annotation'].items():
			if not(v['function'] in function2_roomnum):
				function2_roomnum[v['function']] = []
			function2_roomnum[v['function']].append(k)

		room_nums = function2_roomnum[room_function]
		obj_trans_pose = (-obj_translation[2], -obj_translation[0], 0)
		obj_map_pose = self.get_map_pose_any_pose(obj_trans_pose)
		
		cur_pose_room = None
		succ=False
		for r, r_map in self.room_dict['room_assignment'].items():
			if r in room_nums and not(r in ['free_space', 'free space']) :
				if r_map[obj_map_pose[0], obj_map_pose[1]]==1:
					cur_pose_room = r
					succ=True
					

		if succ == False:
			planner = FMMPlanner(np.ones((720, 720))) 
			r_name2dist = {}
			for r_num, r_map_ori in self.room_dict['room_assignment'].items():
				print("Going through room fmm eval!")
				if not(r_num in ['free_space', 'free space']) and (r_num in room_nums):
					r_map = copy.deepcopy(r_map_ori)
					selem = skimage.morphology.disk(10)
					r_map = skimage.morphology.binary_dilation(r_map , selem)
					planner.set_multi_goal(r_map)
					dist = planner.fmm_dist[obj_map_pose[0], obj_map_pose[1]]
					if dist < 0.5 * 100 / 5.:
						succ = True
						break

		return grasper_closed and succ
