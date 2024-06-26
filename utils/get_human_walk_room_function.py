import cv2
import pickle
import numpy as np
from skimage.measure import regionprops
import skimage.morphology

import copy
from utils.fmm_planner_llm import FMMPlanner


def get_ahead_point(map_pred, sem_map_pred_channels, room_dict, human_traj_index):
	import numpy as np
	human_traj_map = sem_map_pred_channels[human_traj_index].cpu().numpy()
	most_recent_step = int(human_traj_map.max())
	N = 30
	start_step = most_recent_step - N
	if start_step <=0:
		return None
	get_centroid_from = (human_traj_map==most_recent_step).astype(np.uint8)
	properties = regionprops(get_centroid_from, get_centroid_from)
	end_center_of_mass = properties[0].centroid

	#Get the largest 
	human_traj_map2 = copy.deepcopy(human_traj_map)
	human_traj_map2[np.where(human_traj_map > start_step)] = 0.0
	start_step = int(human_traj_map2.max())

	get_centroid_from = (human_traj_map==start_step).astype(np.uint8)
	properties = regionprops(get_centroid_from, get_centroid_from)
	start_center_of_mass = properties[0].centroid

	start_center_of_mass = (int(start_center_of_mass[0]), int(start_center_of_mass[1]))
	end_center_of_mass = (int(end_center_of_mass[0]), int(end_center_of_mass[1]))

	##############################
	#1. Add this vector for 1 meter 
	start_center_of_mass = np.array(start_center_of_mass)
	end_center_of_mass = np.array(end_center_of_mass)

	direction = end_center_of_mass - start_center_of_mass
	unit_direction = direction / np.linalg.norm(direction)
	length = 20
	# Calculate the new end point by extending the vector
	#TODO: check new_end_point inside the same label 
	new_end_point = end_center_of_mass + unit_direction * length
	return new_end_point

def get_human_walk_towards_rooms(map_pred, sem_map_pred_channels, room_dict, human_traj_index, threshold=0.0):
	import numpy as np
	human_traj_map = sem_map_pred_channels[human_traj_index].cpu().numpy()
	most_recent_step = int(human_traj_map.max())
	N = 30
	start_step = most_recent_step - N
	if start_step <=1:
		return [], np.zeros(human_traj_map.shape)
	
	get_centroid_from = (human_traj_map==most_recent_step).astype(np.uint8)
	properties = regionprops(get_centroid_from, get_centroid_from)
	end_center_of_mass = properties[0].centroid

	#Get the largest 
	human_traj_map2 = copy.deepcopy(human_traj_map)
	human_traj_map2[np.where(human_traj_map > start_step)] = 0.0
	start_step = int(human_traj_map2.max())
	if start_step == 0:
		return [], np.zeros(human_traj_map.shape)

	get_centroid_from = (human_traj_map==start_step).astype(np.uint8)
	properties = regionprops(get_centroid_from, get_centroid_from)
	start_center_of_mass = properties[0].centroid

	start_center_of_mass = (int(start_center_of_mass[0]), int(start_center_of_mass[1]))
	end_center_of_mass = (int(end_center_of_mass[0]), int(end_center_of_mass[1]))

	##############################
	#1. Add this vector for 1 meter 
	start_center_of_mass = np.array(start_center_of_mass)
	end_center_of_mass = np.array(end_center_of_mass)

	direction = end_center_of_mass - start_center_of_mass
	if np.linalg.norm(direction) <= 10:
		return [], np.zeros(human_traj_map.shape)
	unit_direction = direction / np.linalg.norm(direction)
	length = 20
	# Calculate the new end point by extending the vector
	#TODO: check new_end_point inside the same label 
	new_end_point = end_center_of_mass + unit_direction * length

	selem = skimage.morphology.disk(4)
	map_pred = skimage.morphology.binary_dilation(map_pred, selem) *1.0
	traversible = (map_pred ==0)
	planner = FMMPlanner(traversible)
	keep_new_end_point = False
	if new_end_point[0] >= 0 and new_end_point[0]<map_pred.shape[0]:
		if new_end_point[1] >= 0 and new_end_point[1]<map_pred.shape[1]:
			goal_end_point = np.zeros(traversible.shape)
			goal_end_point[int(end_center_of_mass[0]), int(end_center_of_mass[1])] = 1
			planner.set_multi_goal(goal_end_point)
			if planner.fmm_dist[int(new_end_point[0]), int(new_end_point[1])] != planner.fmm_dist.max():
				keep_new_end_point = True

	if not(keep_new_end_point):
		new_end_point  = end_center_of_mass 

	##############################
	#2. Get line perpendicular and passes through the 1 meter

	import numpy as np

	# Calculate the slope of the vector V
	delta_x = end_center_of_mass[0] - start_center_of_mass[0]
	delta_y = end_center_of_mass[1] - start_center_of_mass[1]
	slope_V = delta_y / delta_x if delta_x != 0 else float('inf')

	# Calculate the slope of the line perpendicular to V
	slope_perpendicular = -1 / slope_V if slope_V != 0 else 0

	#Something is flipped 

	m = slope_perpendicular
	x1, y1 = new_end_point
	a, b, c = m, -1, -m * new_end_point[0] + new_end_point[1] #-1, y1 - m * x1

	side_of_start = a * start_center_of_mass[0] + b * start_center_of_mass[1] + c

	# Create meshgrid of x and y coordinates
	#xx, yy = np.meshgrid(np.arange(720), np.arange(720))
	xx = np.arange(720)
	yy = np.arange(720)

	side_of_points = a * xx[np.newaxis, :] + b * yy[:, np.newaxis] + c
	side_of_points = side_of_points.T


	# Mark points on the opposite side of start_center_of_mass as 1
	# We use np.sign to determine the side and np.not_equal to check opposite sides
	result_array = np.not_equal(np.sign(side_of_points), np.sign(side_of_start)).astype(int)
	result_array_viz = copy.deepcopy(result_array) * 1.0

	# The array now contains 1s for points on the opposite side of L
	
	blank_array = np.zeros(result_array.shape)
	blank_array[start_center_of_mass[0], start_center_of_mass[1]] = 1
	selem = skimage.morphology.disk(10)
	blank_array = skimage.morphology.binary_dilation(blank_array, selem)
	result_array_viz[np.where(blank_array)] = 0.2

	blank_array = np.zeros(result_array.shape)
	blank_array[int(new_end_point[0]), int(new_end_point[1])] = 1
	selem = skimage.morphology.disk(10)
	blank_array = skimage.morphology.binary_dilation(blank_array, selem)
	result_array_viz[np.where(blank_array)] = 0.8


	#########################
	#3. Get rooms that fall in "1"
	result_array_viz[np.where(map_pred)] = 0.5


	#3-1: "Dilate 1 "in this plane by Multiply traversible as somewhere that is "1" in this plane, and connected to the new_end_point
	
	selem = skimage.morphology.disk(2)
	result_array = skimage.morphology.binary_dilation(result_array*1.0, selem)
	result_array = result_array*1.0 
	assert result_array[int(new_end_point[0]), int(new_end_point[1])] == 1

	traversible = (map_pred ==0) * (result_array == 1)
	planner = FMMPlanner(traversible)
	goal_end_point = np.zeros(traversible.shape)
	goal_end_point[int(new_end_point[0]), int(new_end_point[1])] = 1
	planner.set_multi_goal(goal_end_point)

	valid_room_regions = planner.fmm_dist != planner.fmm_dist.max()

	#3-2: Go through room assignments and get rooms that get 1. 
	valid_rooms = []
	for k, v in room_dict['room_assignment'].items():
		if k!='free_space' and k!='free space':
			if np.sum(v * valid_room_regions) > threshold * np.sum(v * (map_pred ==0))  :
				valid_rooms.append(k)
	return valid_rooms, result_array_viz
