import cv2
import numpy as np


def get_contour_points(pos, origin, size=20):
    x, y, o = pos
    pt1 = (int(x) + origin[0],
           int(y) + origin[1])
    pt2 = (int(x + size / 1.5 * np.cos(o + np.pi * 4 / 3)) + origin[0],
           int(y + size / 1.5 * np.sin(o + np.pi * 4 / 3)) + origin[1])
    pt3 = (int(x + size * np.cos(o)) + origin[0],
           int(y + size * np.sin(o)) + origin[1])
    pt4 = (int(x + size / 1.5 * np.cos(o - np.pi * 4 / 3)) + origin[0],
           int(y + size / 1.5 * np.sin(o - np.pi * 4 / 3)) + origin[1])

    return np.array([pt1, pt2, pt3, pt4])


def draw_line(start, end, mat, steps=25, w=1):
    for i in range(steps + 1):
        x = int(np.rint(start[0] + (end[0] - start[0]) * i / steps))
        y = int(np.rint(start[1] + (end[1] - start[1]) * i / steps))
        mat[x - w:x + w, y - w:y + w] = 1
    return mat


def init_vis_image(goal_inst):
    vis_image = np.ones((620, 1165, 3)).astype(np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (20, 20, 20)  # BGR
    thickness = 2
    goal_thickness = 4
    goal_fontScale = 1.3


    if goal_inst is None:
        goal_text = "Exploration Phase"
    else:
        goal_text = "Goal: {}".format(goal_inst.replace('_', ' ').split('.')[0])
    textsize = cv2.getTextSize(goal_text, font, goal_fontScale, goal_thickness)[0]
    textX = (1150 - textsize[0]) // 2 + 15
    textY = (50 + textsize[1]) // 2
    vis_image = cv2.putText(vis_image, goal_text, (textX, textY),
                            font, goal_fontScale, color, goal_thickness,
                            cv2.LINE_AA)
    y_offset = 75 

    text = "Ego Observations"
    textsize = cv2.getTextSize(text, font, fontScale, thickness)[0]
    textX = (640 - textsize[0]) // 2 + 15
    textY = y_offset + (50 + textsize[1]) // 2
    vis_image = cv2.putText(vis_image, text, (textX, textY),
                            font, fontScale, color, thickness,
                            cv2.LINE_AA)

    text = "Predicted Semantic Map"
    textsize = cv2.getTextSize(text, font, fontScale, thickness)[0]
    textX = 640 + (480 - textsize[0]) // 2 + 30
    textY = y_offset + (50 + textsize[1]) // 2
    vis_image = cv2.putText(vis_image, text, (textX, textY),
                            font, fontScale, color, thickness,
                            cv2.LINE_AA)

    # draw outlines
    color = [100, 100, 100]
    vis_image[49 + y_offset, 15:655] = color
    vis_image[49+ y_offset, 670:1150] = color
    vis_image[50+y_offset:530+y_offset, 14] = color
    vis_image[50+y_offset:530+y_offset, 655] = color
    vis_image[50+y_offset:530+y_offset, 669] = color
    vis_image[50+y_offset:530+y_offset, 1150] = color
    vis_image[530+y_offset, 15:655] = color
    vis_image[530+y_offset, 670:1150] = color


    return vis_image
