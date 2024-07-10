import os
import cv2
from constants import agent_action_prefix, large_objects, small_objects, grabbable_object_categories, color_palette, categories_to_include
import agents.utils.visualization as vu
import copy
import pickle
import numpy as np
from torchvision import transforms
from PIL import Image
import envs.utils.pose as pu

import skimage.morphology

class Perception:
	def __init__(self, args, agent_env):
		self.args = args
		#self.agent = agent
		self.agent_env = agent_env
		self.viz = Visualizer(args)

		self.floorplanner_dict = pickle.load(open('pickles/FP_mc_from_csv.p', 'rb'))
		self.handle_2_category = pickle.load(open('pickles/ycb_ba_g_handle_2_category.p', 'rb'))
		self.res = transforms.Compose(
			[transforms.ToPILImage(),
			 transforms.Resize((args.frame_height, args.frame_width),
							   interpolation=Image.NEAREST)])


	def reset_obj_recep_cat_and_detic(self):
		self.obj_ids_to_handles, self.handles_to_catergories, self.obj_ids_to_complete_handles, self.grabbable_obj_ids, self.grabbable_obj_ids_cur_cat_only = self.get_obj_id_2_handle()
		self.handle2objid = {}
		for obj_id, handle in self.obj_ids_to_handles.items():
			if not handle in self.handle2objid:
				self.handle2objid[handle] =[]
			self.handle2objid[handle].append(obj_id)
		self.human_entry = 788 
		self.obj_ids_to_handles[self.human_entry] = "human"
		self.handles_to_catergories["human"] = "human"
		self.catergoriess_to_handles = {}
		
		for handle, cat in self.handles_to_catergories.items():
			if not (cat in self.catergoriess_to_handles):
				self.catergoriess_to_handles[cat] = []
			self.catergoriess_to_handles[cat].append(handle)    

		self.num_human_observed = 0

		if not(self.args.gt_sem_seg):
			from habitat.tiffany_utils.detic_helper import setup_cfg, detic_args
			from habitat.tiffany_utils.detic_predictor import VisualizationDemo
			categories_to_include_rev = {v:k for k, v in categories_to_include.items()}
			detic_cfg = setup_cfg()
			detic_args.custom_vocabulary = ",".join([categories_to_include_rev[key] for key in sorted(categories_to_include_rev)])
			detic_args.custom_vocabulary = detic_args.custom_vocabulary.replace('human', 'person').replace('washer_dryer', 'washer').replace('chest_of_drawers', 'drawer')#replace('vase', 'jar')
			self.detic_predictor = VisualizationDemo(detic_cfg, detic_args)


	def get_obj_id_2_handle(self):
		id2handle_dict = {}
		id2complete_handle_dict = {}
		handle2semcat_dict = {}
		grabbable_obj_ids = []
		grabbable_obj_ids_cur_cat_only = []
		rom = self.agent_env._sim.get_rigid_object_manager()
		handles = rom.get_object_handles() 
		for handle in handles:
			obj = rom.get_object_by_handle(handle)
			objid = obj.object_id + self.agent_env._sim.habitat_config.object_ids_start
			id2complete_handle_dict[objid] = copy.deepcopy(handle)
			handle = handle.split('_:')[0]#remove the '_:0000'
			if '_part_' in handle:
				handle = handle.split('_part_')[0]
			if handle in self.handle_2_category:
				semantic_category = self.handle_2_category[handle]
				if semantic_category in grabbable_object_categories: 
					grabbable_obj_ids.append(objid)
					if semantic_category == self.agent_env._sim.get_agent_data(0).articulated_agent._sim.ep_info.sif_params['prompt']['initial']['obj_eval']:
						grabbable_obj_ids_cur_cat_only.append(objid)
			elif handle in self.floorplanner_dict:
				semantic_category = self.floorplanner_dict[handle]['main_category']
			else:
				semantic_category = handle 
			handle2semcat_dict[handle] = semantic_category

			id2handle_dict[objid] = handle
			
		return id2handle_dict, handle2semcat_dict, id2complete_handle_dict, grabbable_obj_ids, grabbable_obj_ids_cur_cat_only
	
	def handle_camera_orientation(self, action, obs):
		if not(action is None) and ('cam_down' in action) and action['cam_down']:
			return self.handle_camera_down(obs)
		return self.handle_camera_up(obs)

	def handle_camera_down(self, obs):
		sem_seg = self.generate_semantic_segmentation(obs, down=True)
		state = self.extract_and_combine_rgb_depth(obs, 'down')
		return sem_seg, state

	def handle_camera_up(self, obs):
		sem_seg = self.generate_semantic_segmentation(obs)
		state = self.extract_and_combine_rgb_depth(obs)
		return sem_seg, state

	def generate_semantic_segmentation(self, obs, down=False):
		if self.args.gt_sem_seg:
			return self.generate_sem_seg_gt(obs, down=down)
		return self.generate_sem_seg_detic(obs, down=down)


	def extract_and_combine_rgb_depth(self, obs, suffix=''):
		rgb_key = f"{agent_action_prefix}rgb{'_down' if suffix == 'down' else ''}"
		depth_key = f"{agent_action_prefix}depth{'_down' if suffix == 'down' else ''}"
		rgb = obs[rgb_key].astype(np.uint8)
		depth = obs[depth_key]
		state = np.concatenate((rgb, depth), axis=2).transpose(2, 0, 1)
		return state


	def generate_sem_seg_detic(self, obs, down=False):
		if down:
			rgb = obs["agent_0_articulated_agent_arm_rgb_down"]
		else:
			rgb = obs["agent_0_articulated_agent_arm_rgb"]
		predictions, visualized_output = self.detic_predictor.run_on_image(image=cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), classes_to_keep=None)

		sem_seg_include = np.zeros((480, 640, self.args.num_sem_categories))
		pred_classes = predictions['instances'].pred_classes.cpu().numpy()
		for pi, p_class in enumerate(pred_classes):
			sem_seg_include[:, :, p_class] += predictions['instances'].pred_masks[pi].cpu().numpy()

		if self.args.observed_human_trajectory:
			if np.sum((sem_seg_include[:, :, self.args.human_sem_index] > 0))>0: 
				self.num_human_observed +=1
				sem_seg_include[:, :, self.args.human_trajectory_index] = sem_seg_include[:, :, self.args.human_sem_index] > 0

		sem_seg_include = (sem_seg_include>0) *1.0
		self.viz.rgb_vis = visualized_output.get_image()
		return sem_seg_include

	def generate_sem_seg_gt(self, obs, down=False): 
		if down:
			panoptic = obs[agent_action_prefix  + "panoptic_look_down"]
		else:
			panoptic = obs[agent_action_prefix  + "panoptic"] #(480, 640, 1)

		unique_ids_cur_frame = np.unique(panoptic)
		sem_seg_include = np.zeros((480, 640, self.args.num_sem_categories))
		for oid in unique_ids_cur_frame:
			if oid == 0:
				sem_seg_include[:, :, 0]  = panoptic[:, :, 0] == oid #wall
			if oid in self.obj_ids_to_handles:
				handle = self.obj_ids_to_handles[oid] 
				sem_cat = self.handles_to_catergories[handle] 
				if sem_cat == "human" and self.args.replay_scripted_actions_phase:
					self.human_seen_scripted = True
				if sem_cat in categories_to_include:
					return_cat_idx = categories_to_include[sem_cat]
					sem_seg_include[:, :, return_cat_idx]  = panoptic[:, :, 0] == oid
				if sem_cat == 'door':
					sem_seg_include[:, :, 0]  += panoptic[:, :, 0] == oid #wall

		
		if self.args.observed_human_trajectory:
			if np.sum((sem_seg_include[:, :, self.args.human_sem_index] > 0))>0: 
				self.num_human_observed +=1
				sem_seg_include[:, :, self.args.human_trajectory_index] = sem_seg_include[:, :, self.args.human_sem_index] > 0
		return sem_seg_include  

	def preprocess_obs(self, obs, sem_seg):
		args = self.args
		obs = obs.transpose(1, 2, 0)
		rgb = obs[:, :, :3]
		depth = obs[:, :, 3:4]
	
		if args.gt_sem_seg:
			sem_seg_pred = sem_seg 
			self.viz.rgb_vis = rgb[:, :, ::-1]
		else:
			sem_seg_pred = sem_seg 

		depth = self._preprocess_depth(depth, args.min_depth, args.max_depth)

		ds = args.env_frame_width // args.frame_width  # Downscaling factor
		if ds != 1:
			rgb = np.asarray(self.res(rgb.astype(np.uint8)))
			depth = depth[ds // 2::ds, ds // 2::ds]
			sem_seg_pred = sem_seg_pred[ds // 2::ds, ds // 2::ds]

		depth = np.expand_dims(depth, axis=2)
		state = np.concatenate((rgb, depth, sem_seg_pred),
							   axis=2).transpose(2, 0, 1)

		return state

	def _preprocess_depth(self, depth, min_d, max_d):
		depth = depth[:, :, 0] * 1
		depth = depth*100.0
		return depth
		

class Visualizer:
	def __init__(self, args):
		self.args = args
		self.rgb_vis = None
		


	def reset(self, goal_inst):
		self.vis_image = vu.init_vis_image(goal_inst)


	def _draw_visited(self, args, last_loc,  start, map_pred, timestep):
		if args.visualize or args.print_images:
			if timestep >0:
				# Get last loc
				last_start_x, last_start_y = last_loc[0], last_loc[1]
				r, c = last_start_y, last_start_x
				last_start = [int(r * 100.0 / args.map_resolution),
							  int(c * 100.0 / args.map_resolution)]
				last_start = pu.threshold_poses(last_start, map_pred.shape)
				self.visited_vis = vu.draw_line(last_start, start,
								 self.visited_vis)

	def _visualize(self, inputs, start_x, start_y, viz_goal, args, episode_id, timestep):
		dump_dir = "{}/dump/{}/".format(args.dump_location,args.exp_name)
		ep_dir = '{}/episodes/eps_{}/'.format(
			dump_dir, episode_id)#self.episode_no)
		if not os.path.exists(ep_dir):
			os.makedirs(ep_dir)

		map_pred = inputs['map_pred']
		exp_pred = inputs['exp_pred']
		_, _, start_o, gx1, gx2, gy1, gy2 = inputs['pose_pred']
		
		sem_map = inputs['sem_map_pred']

		#TODO: Change here
		if args.premapping_fbe_mode or args.task_phase: 
			goal = viz_goal #inputs['goal']
			if not (goal is None):
				if not(args.premapping_fbe_mode ):
					selem = skimage.morphology.disk(4)
					goal_mat = 1 - skimage.morphology.binary_dilation(
						goal, selem) != True


				if args.premapping_fbe_mode :
					goal_mat = self.fbe_goal_mat

				goal_mask = goal_mat == 1
				sem_map[goal_mask] = 4


		sem_map += 5

		no_cat_mask = sem_map == args.num_sem_categories + 4 #20
		map_mask = np.rint(map_pred) == 1
		exp_mask = np.rint(exp_pred) == 1
		vis_mask = self.visited_vis == 1

		sem_map[no_cat_mask] = 0
		m1 = np.logical_and(no_cat_mask, exp_mask)
		sem_map[m1] = 2

		m2 = np.logical_and(no_cat_mask, map_mask)
		sem_map[m2] = 1

		sem_map[vis_mask] = 3

		

		if args.premapping_fbe_mode: 
			stg_mat = np.zeros(sem_map.shape)
			stg_mat[int(self.stg[0]), int(self.stg[1])] = 1
			stg_selem = skimage.morphology.star(7)
			stg_mat = skimage.morphology.binary_dilation(stg_mat, stg_selem)
			stg_mask = stg_mat ==1
			#breakpoint()

			sem_map[stg_mask] = 35

		color_pal = [int(x * 255.) for x in color_palette]
		sem_map_vis = Image.new("P", (sem_map.shape[1],
									  sem_map.shape[0]))
		sem_map_vis.putpalette(color_pal)
		sem_map_vis.putdata(sem_map.flatten().astype(np.uint8))
		sem_map_vis = sem_map_vis.convert("RGB")
		sem_map_vis = np.flipud(sem_map_vis)

		sem_map_vis = sem_map_vis[:, :, [2, 1, 0]]
		sem_map_vis = cv2.resize(sem_map_vis, (480, 480),
								 interpolation=cv2.INTER_NEAREST)
		y_offset = 75
		self.vis_image[50 + y_offset: 50 + self.args.env_frame_height + y_offset, 15: 15 + args.env_frame_width] = self.rgb_vis
		self.vis_image[50+y_offset:530+y_offset, 670:1150] = sem_map_vis

		pos = (
			(start_x * 100. / args.map_resolution)
			* 480 / map_pred.shape[0],
			(map_pred.shape[1] - start_y * 100. / args.map_resolution)
			* 480 / map_pred.shape[1],
			np.deg2rad(-start_o)
		)

		agent_arrow = vu.get_contour_points(pos, origin=(670, 50 + y_offset))
		color = (int(color_palette[11] * 255),
				 int(color_palette[10] * 255),
				 int(color_palette[9] * 255))
		cv2.drawContours(self.vis_image, [agent_arrow], 0, color, -1)


		if args.print_images: # and not(ignore_cases):
			if args.run_full_task:
				if args.replay_fbe_actions_phase:
					folder_name = 'replay_fbe_phase'
				elif args.replay_scripted_actions_phase:
					folder_name = 'replay_script_phase'
				elif args.task_phase:
					folder_name = 'task_phase'
				else:
					raise Exception("What phase is this")

				os_f = '{}/episodes/eps_{}/{}'.format(
					dump_dir, episode_id, folder_name)
				if not(os.path.exists(os_f)):
					os.makedirs(os_f)

				fn = '{}/episodes/eps_{}/{}/Vis-{:04d}.png'.format(
					dump_dir, episode_id, folder_name,timestep)


			if args.task_phase:		
				cv2.imwrite(fn, cv2.resize(self.vis_image, (583, 310)))

