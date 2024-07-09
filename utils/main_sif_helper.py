from collections import deque, defaultdict
import torch
import numpy as np
import os 
import pickle

from envs import make_vec_envs

from constants import human_trajectory_index, human_sem_index


#Initializations

def initialize_seed(args):
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	if args.cuda:
		torch.cuda.manual_seed(args.seed)

def configure_device(args):
	args.device = torch.device("cuda:0" if args.cuda else "cpu")
	return args, args.device 

def initialize_environments(args, device):
	torch.set_num_threads(1)
	envs = make_vec_envs(args)
	obs, infos = envs.reset()
	num_content_scene_episodes = [len(info['content_scene_episodes']) for info in infos]
	max_episodes = max(num_content_scene_episodes)
	args.small_obj_channels = infos[0]['small_obj_channels']
	return envs, obs, infos, max_episodes, args, num_content_scene_episodes

def init_episode_data(num_processes, max_episodes):
	episode_success = [deque(maxlen=max_episodes) for _ in range(num_processes)]
	return episode_success

def setup_semantic_mapping(args, device):
	sem_map_module = Semantic_Mapping(args).to(device)
	sem_map_module.eval()
	return sem_map_module

def reset_human_traj(full_map):
    full_map[:, human_trajectory_index+4, :, :] = 0.0 * full_map[:, human_trajectory_index+4, :, :]
    full_map[:, human_sem_index+4, :, :] = 0.0 * full_map[:, human_sem_index+4, :, :] 
    return full_map


#Process tasks, Step action codes
def process_tasks(obs, envs, num_content_scene_episodes, infos, done, episode_success, finished, max_episodes, args, num_scenes, full_map, full_pose, planner_pose_inputs, origins, lmb, tool_policy, baseline_type, device, sem_map_module, step):
	for e, x in enumerate(done):
		if x:
			view_angles = [0.0 for e in range(num_scenes)]
			sem_map_module.set_view_angles(view_angles)

			success = infos[e]['success']
			episode_success[e].append(success)
			if len(episode_success[e]) == num_content_scene_episodes[e]: #3  #num_content_scene_episodes[e]:#num_episodes:
				finished[e] = 1

			full_map, full_pose, locs, planner_pose_inputs, origins, lmb = init_map_and_pose(args,num_scenes, full_map, full_pose, planner_pose_inputs, origins, lmb)

	for e, info in enumerate(infos):
		if info['reset_phase2']:
			#delete human pose
			full_map = reset_human_traj(full_map)

			obs, infos  = envs.reset_phase2()

			#Just Load pose for the map
			full_pose = adjust_full_pose_phase2(args, infos)


	obs, full_map, full_pose, locs, planner_pose_inputs, done, infos = step_actions(
		 args, obs, envs, tool_policy, planner_pose_inputs, infos, num_scenes, full_map, full_pose, lmb, baseline_type, device, sem_map_module, step
	)
	return obs, full_map, full_pose, locs, planner_pose_inputs, done, infos


def step_actions(args, obs, envs, tool_policy, planner_pose_inputs, infos, num_scenes, full_map, full_pose, lmb, baseline_type, device, sem_map_module, step):
	if step==0:
			policy_input_dicts = get_policy_input_dict(args, infos, num_scenes, full_map, full_pose, lmb, baseline_type, step=0, initial=True) 
	else: 
		policy_input_dicts = get_policy_input_dict(args, infos, num_scenes, full_map, full_pose, lmb, baseline_type, step=step, initial=False)

	# ------------------------------------------------------------------

	# Semantic Mapping Module
	poses = torch.from_numpy(np.asarray([infos[env_idx]['sensor_pose'] for env_idx in range(num_scenes)])).float().to(device)

	#Set people as not obstacles
	camera_heights = torch.from_numpy(np.asarray([infos[env_idx]['camera_height'] for env_idx in range(num_scenes)])).float().to(device)
	_, full_map, _, full_pose = \
		sem_map_module(obs, poses, full_map, full_pose, infos[0]['time'], camera_heights = camera_heights, premap_fbe = args.premapping_fbe_mode or args.replay_fbe_actions_phase)#, #human_pose_on_map)


	locs = full_pose.cpu().numpy()
	planner_pose_inputs[:, :3] = locs 

   
	if args.premapping_fbe_mode :
		policy_result_dicts = [{'fbe_goal': 1 - np.rint(full_map[e, 1, :, :].cpu().numpy()) } for e in range(num_scenes)]
	else:
		policy_result_dicts = [tool_policy.execute(policy_input_dicts[e]) for e in range(num_scenes)]

	# Take action and get next observation
	planner_inputs = get_planner_inputs(args, full_map, full_pose, planner_pose_inputs, policy_input_dicts, policy_result_dicts, infos, num_scenes)
	obs, _, done, infos = envs.plan_act_and_preprocess(planner_inputs)


	full_map = update_map_pickup(num_scenes, infos, full_map)
	#Set view_angle 
	view_angles = [infos[e]['view_angle'] for e in range(num_scenes)]
	sem_map_module.set_view_angles(view_angles)
	full_map = location_channel_update(num_scenes, args, full_map, locs)
	return obs, full_map, full_pose, locs, planner_pose_inputs, done, infos




###Map initialization

def initialize_map_and_pose(args, num_scenes, device):
	nc = args.num_sem_categories + 4  # num channels

	# Calculating full and local map sizes
	map_size = args.map_size_cm // args.map_resolution
	full_w, full_h = map_size, map_size

	# Initializing full and local map
	full_map = torch.zeros(num_scenes, nc, full_w, full_h).float().to(device)

	# Initial full and local pose
	full_pose = torch.zeros(num_scenes, 3).float().to(device)

	# Origin of local map
	origins = np.zeros((num_scenes, 3))

	# Local Map Boundaries
	lmb = np.array([[0, 720, 0, 720]])#np.zeros((num_scenes, 4)).astype(int)

	planner_pose_inputs = np.zeros((num_scenes, 7))

	return full_map, full_pose, planner_pose_inputs, origins, lmb

def init_map_and_pose(args, num_scenes, full_map, full_pose, planner_pose_inputs, origins, lmb):
	full_map.fill_(0.)
	full_pose.fill_(0.)
	full_pose[:, :2] = args.map_size_cm / 100.0 / 2.0

	locs = full_pose.cpu().numpy()
	planner_pose_inputs[:, :3] = locs
	for e in range(num_scenes):
		r, c = locs[e, 1], locs[e, 0]
		loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
						int(c * 100.0 / args.map_resolution)]

		full_map[e, 2:4, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1.0


		planner_pose_inputs[e, 3:] = [0., 720.,   0., 720.]#lmb[e]
		origins[e] = [lmb[e][2] * args.map_resolution / 100.0,
					  lmb[e][0] * args.map_resolution / 100.0, 0.]


	return full_map, full_pose, locs, planner_pose_inputs, origins, lmb



###


def get_policy_input_dict(args, infos, num_scenes, full_map, full_pose, lmb, baseline_type, step=0, initial=False):
	if initial:
		policy_input_dicts = [{'time': 0, 
								"task_type": infos[e]['task_type'],
								'human_walking_to_room': infos[e]['human_walking_to_room'],
								'follow_structure': args.follow_structure ,
								'human_follow_stopped': infos[e]['human_follow_stopped'],
								'failed_human_spawn': infos[e]['failed_human_spawn'],
								'human_towards_rooms': infos[e]['human_towards_rooms'],
								'human_pose_room_cur': infos[e]['human_pose_room_cur'],
								'human_pose_room_initial': infos[e]['human_pose_room_initial'],
								'ignore_cur_ep': infos[0]['ignore_cur_ep'],
								'teleport': args.teleport_nav,
								'pddl_list': infos[0]['pddl_list'],
								'execute_count': infos[e]['execute_count'],
								'ep_idx': infos[e]['episode_id'],
								'episode_json': infos[e]['episode_json'],
								'jsons_dir': infos[e]['jsons_dir'],
								"map_pose_human": None,
								"map_pose_agent": None,
								'put_log': infos[e]['put_log'],
								'task_phase': infos[e]['task_phase'],
								'enviornment change': False,
								'new_agent_status': False,
								'is_holding': None,
								'prompt': infos[e]['ep_prompt'],
								'fbe_map_lo_so_dict': infos[e]['fbe_map_lo_so_dict'],
								'scripted_map_lo_so_dict': infos[e]['scripted_map_lo_so_dict'],
								'large_objects': infos[e]['large_objects'],
								'small_objects': infos[e]['small_objects'],
								'human_index':args.human_sem_index,
								'full_map': full_map[e],
								'full_pose': full_pose[e],
								'last_tool_ended': True,
								'room_dict': infos[e]['room_dict'],
								'lmb': lmb[e],
								'human_seen_scripted': infos[e]['human_seen_scripted'],
								'baseline_type': baseline_type,
								'seed': e} for e in range(num_scenes)]

	else:
		policy_input_dicts = [{'time': infos[e]['time'],
								'task_type': infos[e]['task_type'],
								'human_walking_to_room': infos[e]['human_walking_to_room'],
								'follow_structure': args.follow_structure , 
								'human_follow_stopped': infos[e]['human_follow_stopped'],
								'failed_human_spawn': infos[e]['failed_human_spawn'],
								'human_towards_rooms': infos[e]['human_towards_rooms'],
								'human_pose_room_cur': infos[e]['human_pose_room_cur'],
								'human_pose_room_initial': infos[e]['human_pose_room_initial'],
								'ignore_cur_ep': infos[e]['ignore_cur_ep'],
								'pddl_list': infos[e]['pddl_list'],
								'failed_interact': infos[e]['failed_interact'],
								'execute_count': infos[e]['execute_count'],
								'teleport': args.teleport_nav,
								'ep_idx': infos[e]['episode_id'],
								'episode_json': infos[e]['episode_json'],
								'jsons_dir': infos[e]['jsons_dir'],
								"map_pose_human": infos[e]['map_pose_human'],
								"map_pose_agent": infos[e]['map_pose_agent'],
								'prompt': infos[e]['ep_prompt'],
								'put_log': infos[e]['put_log'],
								'enviornment change': False,
								'new_agent_status': False,
								'task_phase': infos[e]['task_phase'],
								'is_holding': infos[e]['holding'],
								'fbe_map_lo_so_dict': infos[e]['fbe_map_lo_so_dict'],
								'scripted_map_lo_so_dict': infos[e]['scripted_map_lo_so_dict'],
								'large_objects': infos[e]['large_objects'],
								'small_objects': infos[e]['small_objects'],
								'human_index':args.human_sem_index,
								'full_map': full_map[e],
								'full_pose': full_pose[e],
								'last_tool_ended': infos[e]['intermediate_stop'],
								'room_dict': infos[e]['room_dict'],
								'lmb': lmb[e],
								'human_seen_scripted': infos[e]['human_seen_scripted'],
								'baseline_type': baseline_type,
								'seed': step*100 + e} for e in range(num_scenes)]
	return policy_input_dicts

def get_planner_inputs(args, local_map, local_pose, planner_pose_inputs, policy_input_dicts, policy_result_dicts, infos, num_scenes):
	planner_inputs = [{} for e in range(num_scenes)]
	for e, p_input in enumerate(planner_inputs):
		#p_input['lmb'] = lmb[e]
		p_input['map_pred'] = local_map[e, 0, :, :].cpu().numpy()
		p_input['exp_pred'] = local_map[e, 1, :, :].cpu().numpy()
		p_input['sem_map_pred_channels'] = local_map[e, 4:, :, :]
		p_input['pose_pred'] = planner_pose_inputs[e]
		p_input['cur_tools'] = policy_result_dicts[e] 
		p_input['wait'] = False 
		p_input['input_dict'] = policy_input_dicts[e]
		if args.visualize or args.print_images:
			local_map[e, -1, :, :] = 1e-5
			p_input['sem_map_pred'] = local_map[e, 4:, :,:].argmax(0).cpu().numpy()
	return planner_inputs



### Policy initialization
def get_tool_policy(args):
	args.follow_structure = None
	if args.follow_human_then_llm:
		args.follow_structure = 'follow_first_then_llm'
	elif args.no_follow_llm:
		args.follow_structure = 'no_follow_llm'
	else: 
		pass
	

	baseline_type ='premapping_or_replay'
	if args.oracle_baseline:
		from utils.oracle_pddl_policy import OraclePDDLPolicy
		tool_policy = OraclePDDLPolicy(teleport=args.teleport_nav)
		baseline_type =  'oracle'
	elif args.follow_human_baseline:
		from utils.follow_human_policy import FollowHumanPolicy
		tool_policy = FollowHumanPolicy(teleport=args.teleport_nav)
		baseline_type =  'follow_human'
	elif args.prompter_baseline:
		from utils.prompter_policy_function import PrompterPolicy
		tool_policy = PrompterPolicy(llm_type =args.llm_type, host=args.llama_host, baseline_type = baseline_type, teleport=args.teleport_nav, magic_grasp = args.magic_grasp_and_put, human_trajectory_index=args.human_trajectory_index, add_human_loc=args.add_human_loc)
	else:
		from utils.llm_policy_function import Reasoner 
		baseline_type = 'reasoning'
		tool_policy = Reasoner(llm_type ='openai_chat', host=args.llama_host, baseline_type = baseline_type, teleport=args.teleport_nav, magic_grasp = args.magic_grasp_and_put, human_recep_no_put=args.human_recep_no_put) #HeuristicPolicyVer1()

	return tool_policy, baseline_type

####Adjust pose for phase 2 
def adjust_full_pose_phase2(args, infos):
	premap_load_folder = "task_load_folder"
	save_folder = os.path.join(premap_load_folder, infos[0]['jsons_dir'].split('/')[-1], infos[0]['episode_json'], infos[0]['episode_id'])
	
	load_from_json_name = infos[0]['load_from_json_name'].split('/')[-1].replace('.json.gz', '')
	load_from_ep_idx = str(infos[0]['load_from_ep_idx'])
	save_folder = os.path.join(premap_load_folder, infos[0]['jsons_dir'].split('/')[-1], load_from_json_name, load_from_ep_idx)
	
	#
	pose_dict = pickle.load(open(os.path.join(save_folder, 'local_pose.p'), 'rb'))
	local_pose = pose_dict['local_pose'].unsqueeze(0)
	full_pose = local_pose.clone()

	return full_pose


###Map update
def update_map_pickup(num_scenes, infos, full_map):
	for e in range(num_scenes):
		if infos[e]['channel_to_remove'] != None:
			channel_idx_2_remove = infos[e]['channel_to_remove'][0]
			wheres_2_remove = infos[e]['channel_to_remove'][1]
			full_map[e, 4+channel_idx_2_remove, :, :][np.where(wheres_2_remove)] = 0.0
	return full_map

def location_channel_update(num_scenes, args, full_map, locs):
	full_map[:, 2, :, :].fill_(0.)  # Resetting current location channel
	for e in range(num_scenes):
		r, c = locs[e, 1], locs[e, 0]
		loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
						int(c * 100.0 / args.map_resolution)]
		full_map[e, 2:4, loc_r - 2:loc_r + 3, loc_c - 2:loc_c + 3] = 1.
	return full_map