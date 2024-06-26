import argparse
import torch


def get_args():
    parser = argparse.ArgumentParser(
        description='Goal-Oriented-Semantic-Exploration')

    parser.add_argument("--task_config", type=str,
                        required=True,
                        help="path to config yaml containing task information")

    parser.add_argument('--gt_sem_seg', action='store_true', default=False) 
    parser.add_argument('--prompter_baseline', action='store_true', default=False) 
    parser.add_argument('--threshold_human_walk',  type=float, default=0.5) 
    parser.add_argument('--human_recep_no_put', action='store_true', default=False)
    parser.add_argument('--add_human_loc', action='store_true', default=True)
    parser.add_argument('--log_oracle_pddl', action='store_true', default=False) 



    parser.add_argument('--oracle_baseline', action='store_true', default=False) 
    parser.add_argument('--oracle_baseline_follow_first', action='store_true', default=False) 
    parser.add_argument('--follow_human_baseline', action='store_true', default=False) 
    parser.add_argument('--follow_human_then_llm', action='store_true', default=False) 
    parser.add_argument('--no_follow_llm', action='store_true', default=False) 

    parser.add_argument('--magic_grasp_and_put', action='store_true', default=True)  
    parser.add_argument('--magic_grasp_threshold',  type=float, default=2.0)
    parser.add_argument('--magic_put_threshold',  type=float, default=2.0) 

    parser.add_argument('--magic_man_if_exists', action='store_true', default=False)  
    parser.add_argument('--eps_to_run', type=str, default="", help='A list of numbers (optional)') 

    parser.add_argument('--human_sem_index', type=int, default=21)
    parser.add_argument('--human_trajectory_index', type=int, default=22)
    parser.add_argument('--observed_human_trajectory', action='store_true', default=True)

    parser.add_argument('--hc_offset', type=float, default=-0.25) 
    parser.add_argument('--proj_exp', action='store_true', default=True)

    parser.add_argument('--premapping_fbe_mode', action='store_true', default=False) 
    parser.add_argument('--replay_fbe_actions_phase', action='store_true', default=False)
    parser.add_argument('--task_phase', action='store_true', default=False)
    parser.add_argument('--run_full_task', action='store_true', default=False) 
    parser.add_argument('--run_task_from_dir', type=str, default="") 

    parser.add_argument('--fbe_maps_save_dir', type=str, default="fbe_maps") 
    parser.add_argument('--llm_type', type=str, default="openai") 
    parser.add_argument('--llama_host', type=str, default="") 
    parser.add_argument('--teleport_nav', action='store_true', default=False) 



    # General Arguments
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--total_num_scenes', type=str, default="auto")
    parser.add_argument('-n', '--num_processes', type=int, default=1,
                        help="""only 1 is supported""")
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument("--sim_gpu_id", type=int, default=0,
                        help="gpu id on which scenes are loaded")
    parser.add_argument("--sem_gpu_id", type=int, default=-1,
                        help="""gpu id for semantic model,
                                -1: same as sim gpu, -2: cpu""")

    # Logging, loading models, visualization
    parser.add_argument('-d', '--dump_location', type=str, default="./tmp/",
                        help='path to dump models and log (default: ./tmp/)')
    parser.add_argument('--exp_name', type=str, default="exp1",
                        help='experiment name (default: exp1)')
    parser.add_argument('-v', '--visualize', type=int, default=0,
                        help="""1: Render the observation and
                                   the predicted semantic map,
                                2: Render the observation with semantic
                                   predictions and the predicted semantic map
                                (default: 0)""")
    parser.add_argument('--print_images', type=int, default=0,
                        help='1: save visualization as images')

    # Environment, dataset and episode specifications
    parser.add_argument('-efw', '--env_frame_width', type=int, default=640,
                        help='Frame width (default:640)')
    parser.add_argument('-efh', '--env_frame_height', type=int, default=480,
                        help='Frame height (default:480)')
    parser.add_argument('-fw', '--frame_width', type=int, default=160,
                        help='Frame width (default:160)')
    parser.add_argument('-fh', '--frame_height', type=int, default=120,
                        help='Frame height (default:120)')

    

    parser.add_argument('--camera_height', type=float, default=0.737158,
                        help="agent camera height in metres")
    parser.add_argument('--hfov', type=float, default=79.0,
                        help="horizontal field of view in degrees")
    parser.add_argument('--turn_angle', type=float, default=30, 
                        help="Agent turn angle in degrees")
    parser.add_argument('--min_depth', type=float, default=0.0,
                        help="Minimum depth for depth sensor in meters")
    parser.add_argument('--max_depth', type=float, default=10.0,
                        help="Maximum depth for depth sensor in meters")
    parser.add_argument('--success_dist', type=float, default=1.0,
                        help="success distance threshold in meters")
    parser.add_argument('--floor_thr', type=int, default=50,
                        help="floor threshold in cm")
    parser.add_argument('--min_d', type=float, default=1.5,
                        help="min distance to goal during training in meters")
    parser.add_argument('--max_d', type=float, default=100.0,
                        help="max distance to goal during training in meters")
    parser.add_argument('--version', type=str, default="v1.1",
                        help="dataset version")

    # Model Hyperparameters
    parser.add_argument('--num_sem_categories', type=int, default=24)


    # Mapping
    parser.add_argument('--global_downscaling', type=int, default=1)
    parser.add_argument('--vision_range', type=int, default=200)
    parser.add_argument('--map_resolution', type=int, default=5)
    parser.add_argument('--du_scale', type=int, default=1)
    parser.add_argument('--map_size_cm', type=int, default=3600)
    parser.add_argument('--cat_pred_threshold', type=float, default=5.0)
    parser.add_argument('--map_pred_threshold', type=float, default=1.0)
    parser.add_argument('--exp_pred_threshold', type=float, default=1.0)
    parser.add_argument('--collision_threshold', type=float, default=0.01) 

    # parse arguments
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    args.sem_gpu_id = -2

    #args.num_processes = 1
    print("Num Processes is ", args.num_processes)
    return args
