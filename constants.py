agent_action_prefix = "agent_0_articulated_agent_arm_" 
hc_offset =-0.25

large_objects = ['wall', 'chair', 'bed', 'shelves', 'toilet', 'bench', 'bathtub', 'fridge', 'couch', 'counter', 'table', 'cabinet', 'car']
small_objects = ['basket', 'book', 'bowl', 'cup', 'hat', 'plate', 'shoe', 'stuffed_toy'] 

grabbable_object_categories = ["action_figure"
                                , "basket"
                                , "book"
                                , "bowl"
                                , "candle_holder"
                                , "canister"
                                , "cup"
                                , "hat"
                                , "mouse_pad"
                                , "pencil_case"
                                , "plate"
                                , "shoe"
                                , "soap_dish"
                                , "sponge"
                                , "stuffed_toy"
                                , "sushi_mat"
                                , "tape"
                                , "vase"]

step_action = ('agent_0_base_velocity', 'agent_0_rearrange_stop', 'agent_0_pddl_apply_action', 'agent_0_oracle_nav_with_backing_up_action', 'agent_1_base_velocity', 'agent_1_rearrange_stop', 'agent_1_pddl_apply_action', 'agent_1_oracle_nav_with_backing_up_action', 'agent_0_arm_action')

#categories_to_include = {'wall': 0, 'chair': 1, 'bed': 2, 'shelves': 3, 'toilet': 4, 'bench': 5, 'bathtub': 6, 'fridge': 7, 'couch': 8, 'counter': 9, 'table': 10, 'cabinet': 11, 'basket': 12, 'book': 13, 'bowl': 14, 'cup': 15, 'hat': 16, 'plate': 17, 'shoe': 18, 'stuffed_toy': 19, 'human': 20}
categories_to_include = {'wall': 0, 'chair': 1, 'bed': 2, 'shelves': 3, 'toilet': 4, 'bench': 5, 'bathtub': 6, 'fridge': 7, 'couch': 8, 'counter': 9, 'table': 10, 'cabinet': 11, 'car': 12, 'basket': 13, 'book': 14, 'bowl': 15, 'cup': 16, 'hat': 17, 'plate': 18, 'shoe': 19, 'stuffed_toy': 20, 'human': 21}

local_w = 720

human_sem_index = 21
human_trajectory_index = 22

color_palette = [
    1.0, 1.0, 1.0,
    0.6, 0.6, 0.6,
    0.95, 0.95, 0.95,
    0.96, 0.36, 0.26,
    0.12156862745098039, 0.47058823529411764, 0.7058823529411765,
    0.9400000000000001, 0.7818, 0.66,
    0.9400000000000001, 0.8868, 0.66,
    0.8882000000000001, 0.9400000000000001, 0.66,
    0.7832000000000001, 0.9400000000000001, 0.66,
    0.6782000000000001, 0.9400000000000001, 0.66,
    0.66, 0.9400000000000001, 0.7468000000000001,
    0.66, 0.9400000000000001, 0.8518000000000001,
    0.66, 0.9232, 0.9400000000000001,
    0.66, 0.8182, 0.9400000000000001,
    0.66, 0.7132, 0.9400000000000001,
    0.7117999999999999, 0.66, 0.9400000000000001,
    0.8168, 0.66, 0.9400000000000001,
    0.9218, 0.66, 0.9400000000000001,
    0.9400000000000001, 0.66, 0.8531999999999998,
    0.9400000000000001, 0.66, 0.748199999999999]
