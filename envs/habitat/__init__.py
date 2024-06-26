# Parts of the code in this file have been borrowed from:
#    https://github.com/facebookresearch/habitat-api
import os
import numpy as np
import torch
#from habitat.config.default import get_config as cfg_env
from habitat_baselines.config.default import get_config as cfg_env
from habitat.datasets.pointnav.pointnav_dataset import PointNavDatasetV1
#from habitat import Config, Env, RLEnv, VectorEnv, make_dataset

from agents.sif_agent import SIF_Agent
from .eif_env import EIF_Env

#from .utils.vector_env import VectorEnv
from omegaconf import OmegaConf

#Copied from construc_vector_env.py in new habitat
import os
import random
from typing import TYPE_CHECKING, Any, List, Type, Optional

#from habitat import ThreadedVectorEnv, VectorEnv, logger, make_dataset
from habitat import logger
from habitat.config import read_write
# from habitat.gym import make_gym_from_config

if TYPE_CHECKING:
    from omegaconf import DictConfig

#copied from the file that has make_gym_from_config
#from habitat.utils.env_utils import make_env_fn
import gym

from habitat.gym.gym_wrapper import HabGymWrapper
from habitat import Dataset
#from habitat.core.vector_env import ThreadedVectorEnv, VectorEnv

from .utils.vector_env import VectorEnv, ThreadedVectorEnv


from habitat.config import read_write
from habitat.config.default import get_agent_config



class SIF_GymHabitatEnv(gym.Wrapper):
    """
    A registered environment that wraps a RLTaskEnv with the HabGymWrapper
    to use the default gym API.
    """

    def __init__(
        self, config: "DictConfig", rank, args, dataset: Optional[Dataset] = None,
    ):
        base_env = SIF_Agent(config=config, rank=rank, args=args, dataset=dataset)
        env = HabGymWrapper(env=base_env)
        super().__init__(env)


def make_gym_from_config(args, config: "DictConfig", rank) -> gym.Env:
    """
    From a habitat-lab or habitat-baseline config, create the associated gym environment.
    """
    if "habitat" in config:
        config = config.habitat
    #env_class_name = _get_env_name(config)
    #env_class = get_env_class(env_class_name)
    env_class = SIF_GymHabitatEnv #(config=config, rank=rank, args=args)#, dataset=dataset)#env=Sem_Exp_Env_Agent)
    assert (
        env_class is not None
    ), f"No environment class with name `{env_class_name}` was found, you need to specify a valid one with env_task"
    return make_env_fn(env_class=env_class, config=config, rank=rank, args=args)


#Copied from env_utils.py of new habitat
from typing import TYPE_CHECKING, Type, Union

from habitat.core.env import Env, RLEnv
from habitat.datasets import make_dataset

if TYPE_CHECKING:
    from omegaconf import DictConfig


def make_env_fn(
    env_class: Union[Type[Env], Type[RLEnv]], config: "DictConfig", rank: int, args
) -> Union[Env, RLEnv]:
    r"""Creates an env of type env_class with specified config and rank.
    This is to be passed in as an argument when creating VectorEnv.

    Args:
        config: root exp config that has core env config node as well as
            env-specific config node.
        env_class: class type of the env to be created.

    Returns:
        env object created according to specification.
    """
    if "habitat" in config:
        config = config.habitat
    dataset = make_dataset(config.dataset.type, config=config.dataset)
    env = env_class(args=args, rank=rank, config=config, dataset=dataset)
    env.seed(config.seed)
    return env


def _get_scenes_from_folder(content_dir):
    scene_dataset_ext = ".glb.json.gz"
    scenes = []
    for filename in os.listdir(content_dir):
        if filename.endswith(scene_dataset_ext):
            scene = filename[: -len(scene_dataset_ext) + 4]
            scenes.append(scene)
    scenes.sort()
    return scenes


def overwirte_args_yaml(args, config_env):
    # config_env.SIMULATOR.AGENT_0.SENSORS = agent_sensors

    # # Reseting episodes manually, setting high max episode length in sim
    # config_env.ENVIRONMENT.MAX_EPISODE_STEPS = 10000000
    # config_env.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False

    # config_env.SIMULATOR.RGB_SENSOR.WIDTH = args.env_frame_width
    # config_env.SIMULATOR.RGB_SENSOR.HEIGHT = args.env_frame_height
    # config_env.SIMULATOR.RGB_SENSOR.HFOV = args.hfov
    # config_env.SIMULATOR.RGB_SENSOR.POSITION = [0, args.camera_height, 0]

    # config_env.SIMULATOR.DEPTH_SENSOR.WIDTH = args.env_frame_width
    # config_env.SIMULATOR.DEPTH_SENSOR.HEIGHT = args.env_frame_height
    # config_env.SIMULATOR.DEPTH_SENSOR.HFOV = args.hfov
    # config_env.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH = args.min_depth
    # config_env.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH = args.max_depth
    # config_env.SIMULATOR.DEPTH_SENSOR.POSITION = [0, args.camera_height, 0]

    # # config_env.SIMULATOR.SEMANTIC_SENSOR.WIDTH = args.env_frame_width
    # # config_env.SIMULATOR.SEMANTIC_SENSOR.HEIGHT = args.env_frame_height
    # # config_env.SIMULATOR.SEMANTIC_SENSOR.HFOV = args.hfov
    # # config_env.SIMULATOR.SEMANTIC_SENSOR.POSITION = \
    # #     [0, args.camera_height, 0]

    # config_env.SIMULATOR.TURN_ANGLE = args.turn_angle
    # config_env.DATASET.SPLIT = args.split
    # config_env.DATASET.DATA_PATH = \
    #     config_env.DATASET.DATA_PATH.replace("v1", args.version)
    # config_env.DATASET.EPISODES_DIR = \
    #     config_env.DATASET.EPISODES_DIR.replace("v1", args.version)

    #config_env.= agent_sensors #skip
    #config_env.habitat.environment.max_episode_steps = 10000000 #Later?
    #breakpoint()
    #breakpoint()
    # config_env.habitat.environment.iterator_options.shuffle = False
    # config_env.habitat.simulator.agents.agent_0.sim_sensors.head_rgb_sensor.width =  args.env_frame_width
    # config_env.habitat.simulator.agents.agent_0.sim_sensors.head_rgb_sensor.height =  args.env_frame_height
    # config_env.habitat.simulator.agents.agent_0.sim_sensors.head_rgb_sensor.hfov = int(args.hfov)
    # config_env.habitat.simulator.agents.agent_0.sim_sensors.head_rgb_sensor.position = [0, args.camera_height, 0]

    # config_env.habitat.simulator.agents.agent_0.sim_sensors.head_depth_sensor.width =  args.env_frame_width
    # config_env.habitat.simulator.agents.agent_0.sim_sensors.head_depth_sensor.height =  args.env_frame_height
    # config_env.habitat.simulator.agents.agent_0.sim_sensors.head_depth_sensor.hfov = int(args.hfov)
    # config_env.habitat.simulator.agents.agent_0.sim_sensors.head_depth_sensor.min_depth = args.min_depth
    # config_env.habitat.simulator.agents.agent_0.sim_sensors.head_depth_sensor.max_depth = args.max_depth
    # config_env.habitat.simulator.agents.agent_0.sim_sensors.head_depth_sensor.position = [0, args.camera_height, 0]

    # config_env.habitat.simulator.turn_angle = args.turn_angle
    # #Do not care about the dataset params

    # config_env.habitat_baselines.torch_gpu_id = args.sim_gpu_id
    #gpu id
    return config_env


def convert_dict_to_string_dict(inp_dict):
    new_dict = {}
    for k,v in inp_dict.items():
        new_dict[str(k)] = str(v)
    return new_dict

 
def add_args_yaml(args, config):
    #gpu id here
    #find all the other args used in main.py, sem_exp_thor.py, objectgoalEnv.py
    args_dict = convert_dict_to_string_dict(vars(args))
    config_dict = vars(config)['_content'] 
    config_dict['OGN_args'] = args_dict
    config = OmegaConf.create(config_dict)
    #config['SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID'] = gpu_id Not sure about this
    return config

#Copied from constrct_vector_env.py in new habitat
#Changed one line (make_env_fn=make_gym_from_config)
def construct_envs(
    args, 
    #config: "DictConfig",#can just read configs later
    workers_ignore_signals=False, #: bool = False,
    enforce_scenes_greater_eq_environments= False#: bool = False,
):
    r"""Create VectorEnv object with specified config and env class type.
    To allow better performance, dataset are split into small ones for
    each individual env, grouped by scenes.

    :param config: configs that contain num_environments as well as information
    :param necessary to create individual environments.
    :param workers_ignore_signals: Passed to :ref:`habitat.VectorEnv`'s constructor
    :param enforce_scenes_greater_eq_environments: Make sure that there are more (or equal)
        scenes than environments. This is needed for correct evaluation.

    :return: VectorEnv object created according to specification.
    """
    config = cfg_env(config_path=args.task_config, configs_dir='config')  #cfg_env('../habitat-lab_soyeonm/habitat-baselines/habitat_baselines/config/' +  args.task_config) #cfg_env('../habitat-lab_soyeonm/habitat-baselines/habitat_baselines/config/' +  args.task_config)
    #Overwrite args into config
    if OmegaConf.is_readonly(config):
        OmegaConf.set_readonly(config, False)
    config = overwirte_args_yaml(args, config)
    config.habitat.simulator.turn_angle = 30
    #breakpoint() #ang_speed set to 20 * 3 (20 was for 10 degrees)
    config.habitat.task.actions.agent_0_base_velocity.ang_speed = 60 #20 * 3

    #Video option
    n_agents = len(config.habitat.simulator.agents)
    for agent_i in range(n_agents):
        agent_name = config.habitat.simulator.agents_order[agent_i]
        agent_config = get_agent_config(
            config.habitat.simulator, agent_i
        )

        agent_sensors = agent_config.sim_sensors
        extra_sensors = config.habitat_baselines.eval.extra_sim_sensors
        with read_write(agent_sensors):
            agent_sensors.update(extra_sensors)
        with read_write(config):
            if config.habitat.gym.obs_keys is not None:
                for render_view in extra_sensors.values():
                    if (
                        render_view.uuid
                        not in config.habitat.gym.obs_keys
                    ):
                        if n_agents > 1:
                            config.habitat.gym.obs_keys.append(
                                f"{agent_name}_{render_view.uuid}"
                            )
                        else:
                            config.habitat.gym.obs_keys.append(
                                render_view.uuid)
            config.habitat.simulator.debug_render = True
    #breakpoint()

    #Add args into config
    config = add_args_yaml(args, config)
    OmegaConf.set_readonly(config, True)
    num_environments = args.num_processes #config.habitat_baselines.num_environments 
    #TODO maybe: change this to num_processes in arguments?
    #num_environments = 1 #Just set this as 1 for now
    configs = []
    dataset = make_dataset(config.habitat.dataset.type)
    scenes = config.habitat.dataset.content_scenes
    if "*" in config.habitat.dataset.content_scenes:
        scenes, episodes = dataset.get_scenes_to_load(config.habitat.dataset)
    #breakpoint()

    scenes_2_episodes_dict = {s: [] for s in scenes}

    #Get a dictionary of scene: episode
    for ep in episodes:
        ep_scene = ep.scene_id.split('/')[-1].replace('.json', '')
        if ep_scene in scenes_2_episodes_dict:
            scenes_2_episodes_dict[ep_scene].append(ep.episode_id)
        else:
            raise Exception("How is this possible?")



    if num_environments < 1:
        raise RuntimeError("num_environments must be strictly positive")

    if len(scenes) == 0:
        raise RuntimeError(
            "No scenes to load, multiple process logic relies on being able to split scenes uniquely between processes"
        )

    random.shuffle(scenes)

    scene_splits: List[List[str]] = [[] for _ in range(num_environments)]#; breakpoint()
    if len(scenes) < num_environments:
        msg = f"There are less scenes ({len(scenes)}) than environments ({num_environments}). "
        if enforce_scenes_greater_eq_environments:
            logger.warn(
                msg
                + "Reducing the number of environments to be the number of scenes."
            )
            num_environments = len(scenes)
            scene_splits = [[s] for s in scenes]
        else:
            logger.warn(
                msg
                + "Each environment will use all the scenes instead of using a subset."
            )
        for scene in scenes:
            for split in scene_splits:
                split.append(scene)
    else:
        for idx, scene in enumerate(scenes):
            scene_splits[idx % len(scene_splits)].append(scene)
        assert sum(map(len, scene_splits)) == len(scenes)

    for i in range(num_environments):
        proc_config = config.copy()
        with read_write(proc_config):
            task_config = proc_config.habitat
            task_config.seed = task_config.seed + i
            if len(scenes) > 0:
                task_config.dataset.content_scenes = scene_splits[i]
                episodes_task_config = []
                for s in task_config.dataset.content_scenes:
                    episodes_task_config += [e for e in scenes_2_episodes_dict[s]]

                import numpy as np
                #shuffle 
                np.random.seed(0)
                np.random.shuffle(episodes_task_config)

                OmegaConf.set_struct(task_config.dataset, False)

                task_config.dataset.scenes_2_episodes_dict = scenes_2_episodes_dict
                task_config.dataset.scenes_episodes = episodes_task_config
                #freeze
                OmegaConf.set_struct(task_config.dataset, True)
                

        configs.append(proc_config)
    #import ipdb; ipdb.set_trace()
    #breakpoint()
    #configs[0].habitat.dataset.scenes_episodes# ['12', '8', '9', '16', '27', '26', '33', '2', '30', '15']


    vector_env_cls: Type[Any]
    if int(os.environ.get("HABITAT_ENV_DEBUG", 0)):
        logger.warn(
            "Using the debug Vector environment interface. Expect slower performance."
        )
        vector_env_cls = ThreadedVectorEnv
    else:
        vector_env_cls = VectorEnv
    vector_env_cls = ThreadedVectorEnv
    #breakpoint()
    envs = vector_env_cls(
        make_env_fn=make_gym_from_config,
        #make_env_fn=make_env_fn,
        env_fn_args=tuple((args, c, rank) for rank, c in enumerate(configs)),
        workers_ignore_signals=workers_ignore_signals,
    )

    if config.habitat.simulator.renderer.enable_batch_renderer:
        envs.initialize_batch_renderer(config)

    return envs, len(scenes)

