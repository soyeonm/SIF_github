import json
import skimage.morphology
import habitat

from envs.utils.fmm_planner import FMMPlanner
import envs.utils.pose as pu
from habitat.core.env import Env, RLEnv

import os
import magnum as mn
import numpy as np
import pickle
import copy

import torch
import _pickle as cPickle
from glob import glob

from constants import agent_action_prefix
from envs.habitat.submodules import  Localize, Perception, ResetTask, Retrieve
from habitat.datasets.rearrange.samplers.receptacle import find_receptacles



class EIF_Env(habitat.RLEnv):#
    """The EIF environment class. The class is responsible
    for loading the dataset, generating episodes, and computing evaluation
    metrics.
    """

    def __init__(self, args, rank, config, dataset): 
        self.args = args 
        self.rank = rank 
        super().__init__(config, dataset)

        # Initializations
        self.episode_no = 0
        self.content_scene_episodes = config.dataset.scenes_episodes
        self.info = {}
        self.info['content_scene_episodes'] = self.content_scene_episodes
    
        #Initialize 
        self.perception = Perception(args, self._env)

        self.loc = Localize(args, self._env)
        self.retr = Retrieve(args, self._env, self.loc)
        self.reset_task = ResetTask(self)

        self.eps_saved_in_tmp_dump = [eps.split('/')[-2].replace('eps_', '') for eps in glob('tmp/dump/' + self.args.exp_name + '/episodes/thread_0/*/model_output.p')]
        if self.args.eps_to_run == "":
            self.eps_to_run = None
        else:
            self.eps_to_run = set([str(int(ep.strip("'").strip('"'))) for ep in self.args.eps_to_run.split(',')]) 
    

    def print_log(self, *statements):
        statements = [str(s) for s in statements]
        statements = ['step #: ', str(self.timestep) , ","] + statements
        joined = ' '.join(statements)
        #self.print_log(joined)
        print(joined)
        self.logs.append(joined)



    ##################################################
    #Reset functions

    def reset_receps(self):
        self.receptacle_instances = find_receptacles(self._env._sim)
        self.receptacle_instances = {r.name: r for r in self.receptacle_instances}


    def get_valid_recep_instance(self, entity_name): #e.g. entity_name is 'bathtub'
        possible_recep_instances = []
        for k,v in self.receptacle_instances.items():
            if k.split('.')[0].replace('receptacle_mesh_', '') in self.perception.handle2objid:
                    if self.perception.handles_to_catergories[k.split('.')[0].replace('receptacle_mesh_', '')] == entity_name:
                        possible_recep_instances.append(v)
        return possible_recep_instances
    
    #Reset for task phase
    def reset_phase2(self):
        args = self.args
        self.reset_task.reset_info(phase1=False)
        self.reset_task.reset_info_for_task_phase()        
        

        self.reset_task.reset_prompt_and_room_dict()
        self.reset_task.reset_pose_phase2()


        self.info['ep_prompt']  = self.reset_task.reset_sif_prompt()
        rgbd_state = self.get_rgbd_state(None)
        state = self.perception.preprocess_obs(rgbd_state, self.sem_seg)

        self.info['task_type'] = self.get_task_type()
        self.reset_task.reset_sif_task_info()
        if self.args.oracle_baseline:
            self.info['pddl_list'] = self.reset_task.read_pddl_seq()
        
        return state , self.info 


    #Reset for new episode
    def reset(self, return_info, **kwargs):
        """Resets the environment to a new episode.

        Returns:
            obs (ndarray): RGBD observations (4 x H x W)
            info (dict): contains timestep, pose, goal category and
                         evaluation metric info
        """
        
        
        args = self.args
        self.logs = []
        obs = super().reset(return_info=False, **kwargs)
        self.reset_task.reset_info()
        self.reset_task.reset_episode_json()

        
        self.goal_cat, _, self.task_type =  self.reset_task.get_goal_cat_and_recep()
        self.perception.reset_obj_recep_cat_and_detic()
        self.reset_task.reset_info_obj()
        self.reset_task.reset_prompt_and_room_dict()


        self.reset_task.reset_pose()
        self.reset_receps()


        self.info['ep_prompt']  = self.reset_task.reset_sif_prompt()
        rgbd_state = self.get_rgbd_state(None)
        state = self.perception.preprocess_obs(rgbd_state, self.sem_seg)

        
        self.info['task_type'] = self.get_task_type()
        self.reset_task.reset_sif_task_info()
        self.reset_task.save_or_pass_generate()
        
        return state , self.info 

    

    ##################################################
    #Manipulation Tools

    def _get_grabbable_obj_id(self, entity_name):
        self.args.magic_grasp_threshold = 2.0
        cur_agent_pose = self._env._sim.get_agent_data(0).articulated_agent.ee_transform().translation 

        sim =self.habitat_env.sim
        rom = sim.get_rigid_object_manager()

        possible_obj_ids = self.perception.grabbable_obj_ids
        if self.args.magic_man_if_exists:
            possible_obj_ids = self.perception.grabbable_obj_ids_cur_cat_only
    
        poses = [rom.get_object_by_id(obj_id - self._env._sim.habitat_config.object_ids_start).translation for obj_id in possible_obj_ids]
        dist_list = np.array([self.util.get_top_down_distance(cur_agent_pose, p) for p in poses])
        min_index = dist_list.argmin()
        chosen_object_id = possible_obj_ids[min_index] - self._env._sim.habitat_config.object_ids_start

        min_dist = None
        if self.args.magic_grasp_and_put:
            min_dist = dist_list[min_index]
            if dist_list[min_index] > self.args.magic_grasp_threshold:
                chosen_object_id = None
        return chosen_object_id, min_dist

    def grab_tool(self,  entity_name):
        robot_id = 0
        chosen_object_id, min_dist = self._get_grabbable_obj_id(entity_name)
        grasp_mgr = self.habitat_env.sim.get_agent_data(robot_id).grasp_mgr
        grabbed_entity = None
        if chosen_object_id != None:
            grabbed_entity = self.perception.handles_to_catergories[self.perception.obj_ids_to_handles[chosen_object_id+self._env._sim.habitat_config.object_ids_start]] #e.g. 'book'
            grasp_mgr.snap_to_obj(chosen_object_id)
            self.habitat_env.sim.internal_step(-1)
            if grasp_mgr.is_grasped:
                self.print_log("Grab success!")
                self.info['holding'] = grabbed_entity
                self.grabbed_obj_id = chosen_object_id 

        if not(grasp_mgr.is_grasped):
            self.print_log("Grab Failure!")
            self.print_log("Min dist was ", min_dist, "; magic grab threshold was ",  self.args.magic_grasp_threshold)

        self.grab_log.append({'entity': grabbed_entity, 'intended_entity': entity_name, 'success': grasp_mgr.is_grasped, 'min_dist': min_dist, 'thrshold': self.args.magic_grasp_threshold})
    


    def _get_puttable_obj_id(self, entity_name):
        self.args.magic_grasp_threshold = 2.0
        cur_agent_pose = self._env._sim.get_agent_data(0).articulated_agent.ee_transform().translation  

        sim =self.habitat_env.sim
        rom = sim.get_rigid_object_manager()

        possible_recep_ids = []
        for k,v in self.receptacle_instances.items():
            if k.split('.')[0].replace('receptacle_mesh_', '') in self.perception.handle2objid:
                if self.args.magic_man_if_exists:
                    if self.perception.handles_to_catergories[k.split('.')[0].replace('receptacle_mesh_', '')] == entity_name:
                        possible_recep_ids += self.perception.handle2objid[k.split('.')[0].replace('receptacle_mesh_', '')]
                else:
                    possible_recep_ids += self.perception.handle2objid[k.split('.')[0].replace('receptacle_mesh_', '')]


        poses = [rom.get_object_by_id(obj_id - self._env._sim.habitat_config.object_ids_start).translation for obj_id in possible_recep_ids]
        dist_list =np.array([self.util.get_top_down_distance(cur_agent_pose,  p) for p in poses])
        if len(possible_recep_ids)==0:
            valid_recep_names, valid_receps, valid_recep_cats = [], [], [] 
            min_dist = 100.0
            return valid_recep_names, valid_receps, valid_recep_cats,  min_dist
        min_index = dist_list.argmin()
        if dist_list[min_index] < self.args.magic_put_threshold:
            chosen_recep_id = possible_recep_ids[min_index] - self._env._sim.habitat_config.object_ids_start
            chosen_recep_handle = self.perception.obj_ids_to_handles[chosen_recep_id + self._env._sim.habitat_config.object_ids_start]
        else:
            chosen_recep_handle = None

        valid_recep_names = []; valid_receps = []; valid_recep_cats = []
        for k,v in self.receptacle_instances.items():
            if k.split('.')[0].replace('receptacle_mesh_', '') == chosen_recep_handle:
                valid_recep_names.append(k)
                valid_receps.append(v)
                valid_recep_cats.append(self.perception.handles_to_catergories[k.split('.')[0].replace('receptacle_mesh_', '')])
        min_dist =  dist_list[min_index]

        return valid_recep_names, valid_receps, valid_recep_cats,  min_dist





    def put_tool(self,  entity_name): 
        valid_recep_names, valid_receps, valid_recep_cats, min_dist = self._get_puttable_obj_id(entity_name)
        robot_id = 0
        grasp_mgr = self.habitat_env.sim.get_agent_data(robot_id).grasp_mgr
        put_recep_cat = None
        if len(valid_receps) >0:
            recep_up = mn.Vector3.y_axis(1.0)
            trans_offset = 0.08
            counter = 0
            put_obj_in_panoptic = False
            
            counter = 0
            #This does not matter in the success
            if self.grabbed_obj_id != None:
                while not(put_obj_in_panoptic) and counter <499:
                    np.random.seed(counter)
                    random_index = np.random.choice(len(valid_recep_names))
                    recep_name = valid_recep_names[random_index]
                    recep_instance = valid_receps[random_index]
                    put_recep_cat = valid_recep_cats[random_index]
                    counter +=1

                    target_object_position = recep_instance.sample_uniform_global(self._env._sim, 1.0) + recep_up*trans_offset
                    rom = self._env._sim.get_rigid_object_manager()
                    set_obj = rom.get_object_by_id(self.grabbed_obj_id)
                    set_obj.translation = target_object_position
                    set_obj.angular_velocity = mn.Vector3.zero_init()
                    set_obj.linear_velocity = mn.Vector3.zero_init()
                    self.habitat_env.sim.internal_step(-1)
                    set_obj.angular_velocity = mn.Vector3.zero_init()
                    set_obj.linear_velocity = mn.Vector3.zero_init()
                    obs = self._env._sim._sensor_suite.get_observations(self._env._sim.get_sensor_observations()) 
                    panoptic = obs[agent_action_prefix  + "panoptic"] 
                    unique_ids_cur_frame = set(np.unique(panoptic))
                    put_obj_in_panoptic = self.grabbed_obj_id + self._env._sim.habitat_config.object_ids_start in unique_ids_cur_frame
                
            
            grasp_mgr.desnap_without_snap_constraints(True)
            grasp_mgr.update_object_to_grasp()
            self.habitat_env.sim.internal_step(-1)
        self.put_log.append({'entity': put_recep_cat, 'intended_entity': entity_name, 'success':not(grasp_mgr.is_grasped), 'min_dist': min_dist, 'thrshold': self.args.magic_put_threshold})
        if not(grasp_mgr.is_grasped):
            self.print_log("Put success!")
            self.info['holding'] = None
            self.grabbed_obj_id = None
        else:
            self.print_log("Put Failure!")
            self.print_log("Min dist was ", min_dist, "; magic put threshold was ",  self.args.magic_put_threshold)
        self.info['put_log'] = self.put_log


    def give_human_tool(self, objid):
        if self.retr.human_spot_same_room():
            robot_id = 1 #human
            chosen_object_id = objid
            grasp_mgr = self.habitat_env.sim.get_agent_data(robot_id).grasp_mgr
            
            if chosen_object_id != None:
                grabbed_entity = self.perception.handles_to_catergories[self.perception.obj_ids_to_handles[chosen_object_id+self._env._sim.habitat_config.object_ids_start]] #e.g. 'book'
                grasp_mgr.snap_to_obj(chosen_object_id)
                self.habitat_env.sim.internal_step(-1)
                if grasp_mgr.is_grasped:
                    self.print_log("Human Grab success!")

                robot_grasp_mgr = self.habitat_env.sim.get_agent_data(0).grasp_mgr
                robot_grasp_mgr.desnap_without_snap_constraints(True)
                robot_grasp_mgr.update_object_to_grasp()
                self.habitat_env.sim.internal_step(-1)


    ##################################################

    #Step functions

    def step(self, action):
        self._reset_manipulate_info()
        if self.args.replay_fbe_actions_phase:
            obs, done, rew = self.step_fbe(action=None)
        else:
            obs, done, rew = self.step_task(action)

        state = self._finalize_step(done, action)
        return state, rew, done, self.info 

    def step_fbe(self, action):
        next_pos = self.fbe_repl_visited_pos_list[self.timestep]
        self._env._sim.agents_mgr[0].articulated_agent.sim_obj.transformation = next_pos['agent0_trans'] 
        self._env._sim.agents_mgr[1].articulated_agent.sim_obj.transformation = next_pos['agent1_trans'] 
        self._env._sim.maybe_update_articulated_agent()
        obs = self._env._sim._sensor_suite.get_observations(self._env._sim.get_sensor_observations()) 
        done =False
        rew = 0.0
        return obs, done, rew

    def step_task(self, action):
        try:
            if action == None:
                obs, rew, done = self.get_deafult_obs_rew_done()
            elif ('hl_tool' in action) and (action['hl_tool'] in ['grab_obj', 'put_obj', "give_human"]):
                obs, rew, done = self._step_mainpulation_tools(action) 
            elif ('agent_0_search' in action['action_args']) or ('agent_0_just_rotate' in action['action_args']):
                obs, rew, done = self._step_search_rotate(action)
            else:
                obs, rew, done, _ = super().step(action)
        except:
            breakpoint()
        return obs, rew, done


    #Step helper functions
    def _step_grab_put_obj(self, action):
        if action['hl_tool'] == 'grab_obj':
            self.grab_tool(action['args']) #entity name
            self.grab_tool_called = True
            self.info['failed_interact'] = not(self.grab_log[-1]['success'])
        elif action['hl_tool'] == 'put_obj':
            #breakpoint()
            self.put_tool(action['args'])
            self.put_tool_called = True
            self.info['failed_interact'] = not(self.put_log[-1]['success'])
        self._env._sim.maybe_update_articulated_agent()
        self.habitat_env.sim.internal_step(-1)
        obs = self._env._sim._sensor_suite.get_observations(self._env._sim.get_sensor_observations()) 
        done = False; rew=0.0
        return obs, done, rew

    def _premap_early_exit(self):
        if self.args.premapping_fbe_mode and self._env._sim.get_agent_data(0).articulated_agent.base_pos == np.array([0.0, 0.0, 0.0]):
            self.done_fbe = True

        #End early
        if self.args.premapping_fbe_mode:
            if self.timestep >= 3000:
                self.done_fbe = True


    def _step_mainpulation_tools(self, action):
        if action['hl_tool'] in ['grab_obj', 'put_obj']:
            return self._step_grab_put_obj(action)
        if action['hl_tool'] == "give_human":
            self.give_human_tool(self.grabbed_obj_id)
            obs = self._env._sim._sensor_suite.get_observations(self._env._sim.get_sensor_observations()) 
            return obs, 0.0, False


    def _perform_search_turns(self, action):
        action['agent_0_search_rotate'] = True
        for _ in range(3):  # Perform three rotations
            obs, rew, done, _ = super().step(action)
        self._env._sim.get_agent_data(0).articulated_agent.at_goal_ona = False
        return obs, rew, done

    def _perform_agent_search(self, action):
        obs, rew, done, _ = super().step(action)
        if self.planner.ll_planner.search_turn_count <= 10:
            return self._perform_search_turns(action)
        else:
            self._env._sim.get_agent_data(0).articulated_agent.at_goal_ona = True
            return obs, rew, done

    def _step_search_rotate(self, action):
        if 'agent_0_search' in action['action_args']:
            return self._perform_agent_search(action)
        if 'agent_0_just_rotate' in action['action_args']:
            return super().step(action)

    

    #Reset before step
    def _reset_manipulate_info(self):
        self.info['failed_interact'] = None



    #Update and Finalize step
    def _finalize_step(self, done, action):
        self._premap_early_exit()
        state = self.get_rgbd_state(action)
        self._update_pose()
        self._update_metrics(done)
        self._update_step_info()
        return state

    def _update_pose(self):
        dx, dy, do = self.loc.get_pose_change()
        self.info['sensor_pose'] = [dx, dy, do]
        self.path_length += pu.get_l2_distance(0, dx, 0, dy)

    def _update_metrics(self, done):
        if done:
            spl, success, dist = self.get_metrics()
            self.info.update({
                'distance_to_goal': dist,
                'spl': spl,
                'success': success
            })

    def _update_step_info(self):
        self.just_reset = False
        self.timestep += 1
        self.info.update({
            'time': self.timestep,
            'camera_height': self.get_camera_height()
        })

    ##################################################
    #Get functions

    def get_deafult_obs_rew_done(self):
        obs = self._env._sim._sensor_suite.get_observations(self._env._sim.get_sensor_observations()) 
        rew = 0.0
        done = False
        return obs, rew, done

    def get_camera_height(self):
        agent_sensor = self._env._sim.get_agent_data(0).articulated_agent._sim._sensors["agent_0_articulated_agent_arm_depth"]
        return agent_sensor._sensor_object.node.transformation.translation[1]

    def get_rgbd_state(self, action):
        self._env._sim.maybe_update_articulated_agent()
        obs = self._env._sim._sensor_suite.get_observations(self._env._sim.get_sensor_observations())
        self.sem_seg, state = self.perception.handle_camera_orientation(action, obs)
        if not(action is None) and 'cam_down' in action:
            self.info['view_angle'] = -0.36157828398093383
        else:
             self.info['view_angle'] = 0.0
        return state

    def get_task_type(self):
        initial_prompt = self._env._sim.get_agent_data(0).articulated_agent._sim.ep_info.sif_params['prompt']['initial']
        if self._env._sim.get_agent_data(0).articulated_agent._sim.ep_info.sif_params['prompt']['recep_type'] == 'human':
            task_type ='human'
        else:
            #task_type = 'obj'
            if initial_prompt['recep_eval_unique']!=None or initial_prompt['recep_eval_notunique']!=None:
                task_type = 'recep'
            else:
                task_type = 'room'
        return task_type


    #exists to just run with habitat
    ##################################################
    def get_reward_range(self):
        """This function is not used, Habitat-RLEnv requires this function"""
        return (0., 1.0)

    def get_reward(self, observations):
        return 0.0

    def get_metrics(self):
        return 0.0, 0.0, 0.


    def get_done(self, observations):
        return False


    def get_info(self, observations):
        """This function is not used, Habitat-RLEnv requires this function"""
        info = {}
        return info

    def get_spaces(self):
        """Returns observation and action spaces for the ObjectGoal task."""
        return self.observation_space, self.action_space
    
    

    

