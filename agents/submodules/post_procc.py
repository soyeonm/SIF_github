import os
from constants import categories_to_include, agent_action_prefix
import numpy as np
import skimage.morphology
from envs.utils.fmm_planner import FMMPlanner
from scipy.spatial.transform import Rotation


class PostProcess: 
	def __init__(self, agent):
		self.agent = agent

	def map_update_put_grab(self, planner_inputs, action, put=True, entity_name=None):
		grab = not(put)
		if grab:
			rom = self.agent._env.sim.get_rigid_object_manager()
			if self.agent.grabbed_obj_id !=None:
				channel_idx_2_remove = categories_to_include[self.agent.perception.handles_to_catergories[rom.get_object_by_id(self.agent.grabbed_obj_id).handle.split('_:')[0]]]
				cat_of_interest = self.agent.perception.handles_to_catergories[rom.get_object_by_id(self.agent.grabbed_obj_id).handle.split('_:')[0]]
			else:
				channel_idx_2_remove = categories_to_include[entity_name] 
				cat_of_interest = entity_name
			wheres_channel = planner_inputs['sem_map_pred_channels'][channel_idx_2_remove].cpu().numpy() 
			if np.sum( wheres_channel)>0:
				labeled = skimage.morphology.label((wheres_channel >0 )*1.0)
				labels = np.unique(labeled)
				
				planner = FMMPlanner(np.ones(wheres_channel.shape))
				dist_to_lab_dict = {}
				for label in labels:
					if np.sum(wheres_channel[np.where(labeled==label)]) ==0:
						pass
					else:
						planner.set_multi_goal(labeled ==label)
						dist_to_lab_dict[label] = planner.fmm_dist[self.agent.planner.start_taskphase[0], self.agent.planner.start_taskphase[1]]

				sorted_obj = dict(sorted(dist_to_lab_dict.items()))
				closest_label = list(sorted_obj.keys())[0]
				where_to_remove = labeled == closest_label
				self.agent.info['channel_to_remove'] = (channel_idx_2_remove,  where_to_remove)

		if put: 
			cat_of_interest = action['args']
			put_object_seen = np.sum(self.agent.sem_seg[:, :, categories_to_include[cat_of_interest ]]) >0
			if not(put_object_seen):
				obs = self.agent._env._sim._sensor_suite.get_observations(self.agent._env._sim.get_sensor_observations())
				sem_seg, state = self.agent.perception.handle_camera_orientation(None, obs)
				put_os_new = np.sum(sem_seg[:, :, categories_to_include[cat_of_interest]]) >0
			else:
				put_os_new = put_object_seen 
				#Save 
			obs = self.agent._env._sim._sensor_suite.get_observations(self.agent._env._sim.get_sensor_observations())
			self.agent.info['scripted_put_was_seen'] = put_os_new 
			rgb = obs[agent_action_prefix + "rgb_down"].astype(np.uint8)
			depth = obs[agent_action_prefix  + "depth_down"]
			state = np.concatenate((rgb, depth), axis=2).transpose(2, 0, 1)
			cam_rot_matrix = self.agent._env._sim.get_agent_data(0).articulated_agent._sim._sensors[agent_action_prefix  + "depth_down"]._sensor_object.node.transformation.rotation()
			rot = Rotation.from_matrix(np.array(cam_rot_matrix))
			self.agent.info['view_angle'] =  -0.36157828398093383 
			return state
		return None