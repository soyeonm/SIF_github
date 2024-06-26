import numpy as np
import os

os.environ["OMP_NUM_THREADS"] = "1"

from utils.policy_functions import * #select_random_room_policy
from utils.main_sif_helper import *
from utils.mapper import Semantic_Mapping
from arguments import get_args





'''main.py code to run Reasoner, Prompter, Oracle, or
Your own policy 
on SIF tasks'''

def main():
    args = get_args()

    initialize_seed(args)
    args, device = configure_device(args)
    envs, obs, infos, max_episodes, args,num_content_scene_episodes = initialize_environments(args, device)
    episode_success = init_episode_data(args.num_processes, max_episodes)


    finished = np.zeros((args.num_processes))
    torch.set_grad_enabled(False)
    num_scenes = 1

    full_map, full_pose, planner_pose_inputs, origins, lmb = initialize_map_and_pose(args, num_scenes, device)
    full_map, full_pose, locs, planner_pose_inputs, origins, lmb = init_map_and_pose(args,num_scenes, full_map, full_pose, planner_pose_inputs, origins, lmb)

    # Semantic Mapping
    sem_map_module = Semantic_Mapping(args).to(device) 
    sem_map_module.eval()

    ######################################################################
    #High Level Policy (e.g. LLM)
    #CHANGE HERE (get_tool_policy)
    #to run tasks with your own policy
    #supports Reasoner, Prompter, Oracle policy by default
    tool_policy, baseline_type = get_tool_policy(args)
    ###############################################################
    torch.set_grad_enabled(False)
    done = [False] * num_scenes
    step = 0

    while not all(finished):
        obs, full_map, full_pose, locs, planner_pose_inputs, done, infos = process_tasks(
            obs, envs, num_content_scene_episodes, infos, done, episode_success, finished, max_episodes, args, 
            num_scenes, full_map, full_pose, planner_pose_inputs, origins, lmb, tool_policy, baseline_type, 
            device, sem_map_module, step
        )
        step +=1
        



if __name__ == "__main__":
    main()