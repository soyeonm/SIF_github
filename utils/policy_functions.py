import numpy as np
import pickle
import copy

#From heuristic policy to 
#common sense policy to 
#Reasoning policy (LLM)


###########################
####Heuristic Tool Choosing policy
class HeuristicPolicyVer1:
	def __init__(self):
		#avaialble high level tools
		self.hl_tools = ["nav_to_room", "explore_room", "grab_obj", "put_obj", "END_TASK"]
		self.whether_execute_type = 'last_tool_ended'
		self.end_time = 300
		self.last_lmb = np.array([0,0,0,0])

	def reset_lmb(self, lmb):
		self.last_lmb = lmb

	###########################
	#Choose actions
	def choose_high_level_tool(self, input_dict):
		if input_dict['time'] >= self.end_time:
			return "END_TASK"
		elif input_dict['time'] >=220:
			return "grab_obj"
		else:
			return "nav_to_room"


	def choose_low_level(self, high_level_tool, input_dict):
		if high_level_tool == "nav_to_room":
			return select_random_room_policy(input_dict['seed'], input_dict['room_dict'])

		elif high_level_tool == "grab_obj":
			return '002_master_chef_can'

		elif high_level_tool == "explore_room":
			raise Exception("Not Implemented!")

	###########################
	#Decide actions
	def decide_whether_cur_hl_tool_ended(self, input_dict):
		raise Exception("Not Implemented!")

	def decide_whether_execute(self, input_dict):
		
		task_end_conidition = input_dict['time'] >= self.end_time
		if input_dict['last_tool_ended'] == True or task_end_conidition:
			return True
		else:
			return False


	def adjust_ll_tool(self, ll_argument, input_dict):
		if not(ll_argument is None):
			return [ll_argument[0]-input_dict['lmb'][0], ll_argument[1]-input_dict['lmb'][2]]
		else:
			return None

	

	def execute(self, input_dict):
		whether_execute = self.decide_whether_execute(input_dict)

		if input_dict['task_phase']:
			if whether_execute:
				hl_tool = self.choose_high_level_tool(input_dict)
				ll_argument_before_adjust = self.choose_low_level(hl_tool, input_dict)
				self.last_return_dict = {'hl_tool': hl_tool, 'll_argument_before_adjust': ll_argument_before_adjust, 'whether_execute': whether_execute}
			else:
				self.last_return_dict['whether_execute'] = whether_execute
			
			if self.last_return_dict['hl_tool'] in ["nav_to_room", "explore_room"]:
				self.last_return_dict['ll_argument'] = self.adjust_ll_tool(self.last_return_dict['ll_argument_before_adjust'], input_dict)
			else:
				self.last_return_dict['ll_argument'] = self.last_return_dict['ll_argument_before_adjust']
			return self.last_return_dict
		else:
			self.last_return_dict = {}
			return self.last_return_dict
        





####Heuristic Policies for navigation
def select_random_room_policy(seed, room_dict):
	#randomly choose a room 
	np.random.seed(seed)
	room = np.random.choice(list(room_dict.keys()))
	print("Chose room ", room, "!")
	#randomly chooose a goal inside the room
	wheres = np.where(room_dict[room])
	len_where = len(wheres[0])
	np.random.seed(seed)
	where_idx = np.random.choice(np.arange(len_where))
	#idx in map
	global_goal = [wheres[0][where_idx], wheres[1][where_idx]]
	return global_goal
