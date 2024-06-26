import os
from constants import hc_offset

import copy
import numpy as np
from scipy.spatial.transform import Rotation
import magnum as mn
import quaternion
import torch

import envs.utils.pose as pu


class Localize:
	def __init__(self, args, agent_env):
		self.args = args
		self.agent_env = agent_env




	def reset(self, agentenv, args, starting):
		self.agent_env = agentenv
		self.last_sim_location = self.get_sim_location()
		self.last_ee_location = self.get_ee_cam_location()
		self.agent_last_sim_location = self.get_sim_agent_location()
		self.args = args

		if starting:
			self.agent_0_starting_camera_pose = copy.deepcopy(self.last_sim_location)
			self.agent_0_starting_agent_pose = copy.deepcopy(self.agent_last_sim_location)

			self.rel_o_offset = (self.agent_last_sim_location[2] - self.last_sim_location[2]-np.pi)*180/np.pi
			self.agent_0_starting_camera_pose_in_agent_coordinate = (self.last_ee_location[0], self.last_ee_location[1], self.last_ee_location[2])

	def get_new_pose_batch(self, pose, rel_pose_change):
		import copy
		pose_return = copy.deepcopy(pose)
		offset = 0.0
		pose_return[:, 1] += rel_pose_change[:, 0] * torch.sin(pose[:, 2]-offset / 57.29577951308232) + rel_pose_change[:, 1] *  torch.cos(pose[:, 2]-offset / 57.29577951308232)
		pose_return[:, 0] += rel_pose_change[:, 0] * torch.cos(pose[:, 2]-offset / 57.29577951308232) - rel_pose_change[:, 1] * torch.sin(pose[:, 2]-offset / 57.29577951308232)
		pose_return[:, 2] += rel_pose_change[:, 2] * 57.29577951308232

		pose_return[:, 2] = torch.fmod(pose[:, 2] - 180.0, 360.0) + 180.0
		pose_return[:, 2] = torch.fmod(pose[:, 2] + 180.0, 360.0) - 180.0

		return pose_return



	#0. Agent and Human odometry
	def get_sim_agent_location(self):
		"""Returns x, y, o pose of the agent in the Habitat simulator."""
		agent_state = self.agent_env.sim.get_agent_state(0)
		x = -agent_state.position[2]
		y = -agent_state.position[0]
		axis = quaternion.as_euler_angles(agent_state.rotation)[0]
		if (axis % (2 * np.pi)) < 0.1 or (axis %
										  (2 * np.pi)) > 2 * np.pi - 0.1:
			o = quaternion.as_euler_angles(agent_state.rotation)[1]
		else:
			o = 2 * np.pi - quaternion.as_euler_angles(agent_state.rotation)[1]
		o = o + hc_offset
		if o > np.pi:
			o -= 2 * np.pi
		return x, y, o

	def get_sim_human_location(self):
		"""Returns x, y, o pose of the agent in the Habitat simulator."""
		agent_state = self.agent_env.sim.get_agent_state(1)
		x = -agent_state.position[2]
		y = -agent_state.position[0]
		axis = quaternion.as_euler_angles(agent_state.rotation)[0]
		if (axis % (2 * np.pi)) < 0.1 or (axis %
										  (2 * np.pi)) > 2 * np.pi - 0.1:
			o = quaternion.as_euler_angles(agent_state.rotation)[1]
		else:
			o = 2 * np.pi - quaternion.as_euler_angles(agent_state.rotation)[1]
		o = o + hc_offset
		if o > np.pi:
			o -= 2 * np.pi
		return x, y, o


	def get_ee_cam_location(self):
		"""Returns x, y, o pose of the agent in the Habitat simulator."""

		cam_info = self.agent_env._sim.get_agent_data(0).articulated_agent.params.cameras["agent_0_articulated_agent_arm_depth"]
		link_trans = self.agent_env._sim.get_agent_data(0).articulated_agent.sim_obj.get_link_scene_node(cam_info.attached_link_id).transformation
		# Get the camera offset transformation
		offset_trans = mn.Matrix4.translation(cam_info.cam_offset_pos)
		cam_trans = link_trans @ offset_trans @ cam_info.relative_transform


		cam_translation = cam_trans.translation
		cam_rot_matrix =cam_trans.rotation()


		rot = Rotation.from_matrix(np.array(cam_rot_matrix))

		#Get the Euler angles in radians
		cam_o = rot.as_euler('xyz')[1]
		axis = rot.as_euler('xyz')[0]

		cam_x = cam_translation[2]
		cam_y = cam_translation[0]
		x,y,o = cam_x, cam_y, cam_o
		_, _, o = self.get_sim_agent_location()
		return -x, -y, o


	def get_sim_location(self):
		"""Returns x, y, o pose of the agent in the Habitat simulator."""
		cam_translation = self.agent_env._sim.get_agent_data(0).articulated_agent._sim._sensors["agent_0_articulated_agent_arm_depth"]._sensor_object.node.transformation.translation
		cam_rot_matrix = self.agent_env._sim.get_agent_data(0).articulated_agent._sim._sensors["agent_0_articulated_agent_arm_depth"]._sensor_object.node.transformation.rotation()
		rot = Rotation.from_matrix(np.array(cam_rot_matrix))

		#Get the Euler angles in radians
		cam_o = rot.as_euler('xyz')[1]
		axis = rot.as_euler('xyz')[0]

		cam_x = cam_translation[2]
		cam_y = cam_translation[0]
		if (axis % (2 * np.pi)) < 0.15 or (axis % (2 * np.pi)) > 2 * np.pi - 0.15:
			cam_o = np.pi + cam_o
		else:
			cam_o = 2 * np.pi - cam_o 
		x,y,o = cam_x, cam_y, cam_o
		if o > np.pi:
			o -= 2 * np.pi

		return self.get_ee_cam_location()


	def get_pose_change_ee(self, do_last=True):
		"""Returns dx, dy, do pose change of the agent relative to the last
		timestep."""
		self.agent_env._sim.maybe_update_articulated_agent()
		curr_ee_pose = self.get_ee_cam_location() 
		dx, dy, do = pu.get_rel_pose_change(
			curr_ee_pose, self.last_ee_location)
		self.last_ee_location = curr_ee_pose
		return dx, dy, do

	def get_pose_change(self, do_last=True):
		"""Returns dx, dy, do pose change of the agent relative to the last
		timestep."""
		self.agent_env._sim.maybe_update_articulated_agent()
		curr_sim_pose = self.get_sim_location()
		dx, dy, do = pu.get_rel_pose_change(
			curr_sim_pose, self.last_sim_location)
		self.last_sim_location = curr_sim_pose
		return dx, dy, do