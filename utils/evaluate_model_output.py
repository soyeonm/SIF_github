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
oracle_path_len = pickle.load(open('pickles/oracle_traj_len.p', 'rb'))[args.json_name]


from glob import glob
for eval_path in glob(exp_result_path + '/eps_*/eval_result.p'):
	eval_result = pickle.load(open(eval_path, 'rb'))
	model_output = pickle.load(open(eval_path.replace('eval_result.p', 'model_output.p') , 'rb'))
	eps_num = eval_path.split('/')[-2]
	spl = eval_result * oracle_path_len [eps_num] / max(model_output['steps_taken'], oracle_path_len[int(eps_num.replace('eps_', ''))])
	success_list[int(eps_num.replace('eps_', ''))] = eval_result
	spl_list[int(eps_num.replace('eps_', ''))] = spl
	total_count +=1



#Just see for ambiguous  episodes
print("SR is ", np.mean(list(success_list.values())))
print("SPL is ", np.mean(list(spl_list.values())))

# print("AMB SR is ", np.mean(amb_success_list))
# print("Amb len ", len(amb_success_list))
#print(amb_fail_idxes)
fail_idx = []
for k, v in success_list.items():
	if not(v):
		fail_idx.append(k)
print("Failed idxes are ", fail_idx)
