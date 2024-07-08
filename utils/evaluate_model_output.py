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
	#spl = eval_result * oracle_path_len [eps_num] / max(model_output['steps_taken'], oracle_path_len [eps_num])
	success_list[int(eps_num.replace('eps_', ''))] = eval_result
	#spl_list[int(eps_num.replace('eps_', ''))] = spl
	total_count +=1


task_json_path = 'data/datasets/sif_release/jsons/' +  args.json_name +  '.json.gz' #e.g. 
with gzip.open(task_json_path, 'rb') as f:
	data = f.read()
	# Decode the bytes object to string
	json_str = data.decode('utf-8')
	# Parse the JSON data
	task_json_data = json.loads(json_str)

amb_success_list = []
for ep_idx, episode in enumerate(task_json_data['episodes']):
	ambiguous = episode['sif_params']['ambiguous']
	success = success_list[ep_idx] 
	#spl = spl_list[ep_idx] 
	if ambiguous:
		amb_success_list.append(success)
		#amb_spl_list.append(spl)
		#amb_idxes.append(ep_idx)



#Just see for ambiguous  episodes
print("SR is ", np.mean(list(success_list.values())))
print("AMB SR is ", np.mean(amb_success_list))
print("Amb len ", len(amb_success_list))