import gzip
import json
import argparse
import pickle
import numpy as np



parser = argparse.ArgumentParser(description="")
parser.add_argument('--exp_name',  type=str, required=True) 
parser.add_argument('--json_name',  type=str, required=True)
args = parser.parse_args()

#Just get SR
#Only need exp_name for this 

exp_result_path = 'tmp/dump/' + args.exp_name + '/episodes/thread_0'
success_list = {}
spl_list = {}
total_count = 0
from glob import glob
for eval_path in glob(exp_result_path + '/eps_*/eval_result.p'):
	eval_result = pickle.load(open(eval_path, 'rb'))
	model_output = pickle.load(open(eval_path.replace('eval_result.p', 'model_output.p') , 'rb'))
	eps_num = eval_path.split('/')[-2]
	spl = eval_result * oracle_path_len [eps_num] / max(model_output['steps_taken'], oracle_path_len [eps_num])
	success_list[int(eps_num.replace('eps_', ''))] = eval_result
	spl_list[int(eps_num.replace('eps_', ''))] = spl
	total_count +=1

breakpoint()
#data/datasets/sif_release