import gzip
import json
import argparse
import pickle
import numpy as np



parser = argparse.ArgumentParser(description="")
parser.add_argument('--exp_name',  type=str, required=True) 
parser.add_argument('--json_name',  type=str, required=True)
parser.add_argument('--get_amb_clear', action='store_true', default=False)
args = parser.parse_args()


exp_result_path = 'tmp/dump/' + args.exp_name + '/episodes/thread_0'
success_list = {}
spl_list = {}
total_count = 0
oracle_path_len = pickle.load(open('pickles/oracle_traj_len.p', 'rb'))[args.json_name]


from glob import glob
for eval_path in glob(exp_result_path + '/eps_*/eval_result.p'):
	eval_result = pickle.load(open(eval_path, 'rb'))
	model_output = pickle.load(open(eval_path.replace('eval_result.p', 'model_output.p') , 'rb'))
	eps_num = eval_path.split('/')[-2]
	spl = eval_result * oracle_path_len[int(eps_num.replace('eps_', ''))] / max(model_output['steps_taken'], oracle_path_len[int(eps_num.replace('eps_', ''))])
	success_list[int(eps_num.replace('eps_', ''))] = eval_result
	spl_list[int(eps_num.replace('eps_', ''))] = spl
	total_count +=1



print("SR is ", np.mean(list(success_list.values())))
print("SPL is ", np.mean(list(spl_list.values())))
print("# Tasks ran: ", len(list(success_list.values())))


fail_idx = []
for k, v in success_list.items():
	if not(v):
		fail_idx.append(k)
print("Failed idxes are ", fail_idx)

#Also print by ambiguous/ clear
if args.get_amb_clear:
	amb_success_list = []
	clear_success_list = []
	amb_spl_list = []
	clear_spl_list = []

	clear_idxes = []
	amb_idxes = []

	task_json_path = 'data/datasets/sif_release/jsons/' +  args.json_name +  '.json.gz' #e.g. 
	with gzip.open(task_json_path, 'rb') as f:
		data = f.read()
		json_str = data.decode('utf-8')
		task_json_data = json.loads(json_str)

	for ep_idx, episode in enumerate(task_json_data['episodes']):
		ambiguous = episode['sif_params']['ambiguous']
		success = success_list[ep_idx] 
		spl = spl_list[ep_idx] 
		if ambiguous:
			amb_success_list.append(success)
			amb_spl_list.append(spl)
			amb_idxes.append(ep_idx)
		else:
			clear_success_list.append(success)
			clear_spl_list.append(spl)
			clear_idxes.append(ep_idx)

	print("amb + clear len ", len(amb_success_list + clear_success_list))

	print("clear len ", len(clear_success_list))
	print("amb len ", len(amb_success_list))

	print("CLEAR SR is ", np.mean(clear_success_list))
	print("CLEAR SPL is ", np.mean(clear_spl_list))

	print("AMB SR is ", np.mean(amb_success_list))
	print("AMB SPL is ", np.mean(amb_spl_list))

	print("Amb idxes ", amb_idxes)
	print("Clear idxes ", clear_idxes)	
