	

class IntermediateStopTracker:

	def __init__(self, agent):
		self.agent = agent

	def check_intermediate_stop(self, planner_inputs):
		self.agent.info['intermediate_stop'] = False
		hl_tool = planner_inputs['cur_tools']['hl_tool']
		if 'll_argument' in planner_inputs['cur_tools']:
			ll_argument = planner_inputs['cur_tools']['ll_argument']
		else:
			ll_argument = None
		if hl_tool in ["nav_to_room", "explore_room"]:
			if self._is_at_goal():
				self._set_intermediate_stop("Intermediate stop called!")

		elif hl_tool in ["follow_human", "no_op"] :
			self._set_intermediate_stop_human(planner_inputs, ll_argument)

		else:
			self._set_intermediate_stop_tool_commands(hl_tool)

		if self.agent.info['intermediate_stop'] and hl_tool != "follow_human":
			self.agent._reset_agent_goal()

	def _is_at_goal(self):
		return self.agent._env._sim.get_agent_data(0).articulated_agent.at_goal_ona

	def _set_intermediate_stop(self, log_message):
		self.agent.info['intermediate_stop'] = True
		self.agent.print_log(log_message)

	def _set_intermediate_stop_human(self, planner_inputs, ll_argument):
		if self.agent.timestep == 1:
			self.agent.info['human_pose_room_initial'] = self.agent.retr.get_human_room(planner_inputs)

		if planner_inputs['cur_tools']['hl_tool'] == 'no_op':
			self._set_intermediate_stop("")

		else:
			if  self.agent.timestep % 20 == 0 and ll_argument['ask_again_every_20']:
				self._set_intermediate_stop("")

			if self.agent.timestep > 0 and self.agent.planner._decide_human_follow_stopped() and self._is_at_goal():
				self.agent.info.update({
					'human_follow_stopped': True,
					'intermediate_stop_human_follow': True
				})
				self._set_intermediate_stop('human_follow_stopped declared!')

			if self._is_at_goal():
				self.agent.info['intermediate_stop_human_follow'] = True


			if not self.agent.retr.human_visible(self.agent.sem_seg):
				self.agent.info['intermediate_stop_human_follow'] = True

	def _set_intermediate_stop_tool_commands(self, hl_tool):
		tools_require_stop = {
			"grab_obj": self.agent.grab_tool_called,
			"put_obj": self.agent.put_tool_called,
			"nav_directly_to_goal_pick": self.agent.grab_tool_called,
			"nav_directly_to_goal_put": self.agent.put_tool_called,
			'no_op': True
		}

		if (hl_tool in tools_require_stop) and tools_require_stop[hl_tool]:
			self._set_intermediate_stop("")

		if hl_tool == 'give_human':
			self._set_intermediate_stop("")
