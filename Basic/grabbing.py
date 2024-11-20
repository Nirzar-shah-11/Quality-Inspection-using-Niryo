# from pyniryo import *
# import numpy as np
# import time
# import math
# robot = NiryoRobot("10.10.10.10")
# robot.calibrate_auto()
# robot.calibrate_auto()
# robot.update_tool()
# robot.move_joints(0.5,0.3,0.1,0.57,0.1,0)
# robot.grasp_with_tool()
# robot.release_with_tool()

# # Getting image
# img_compressed = robot.get_img_compressed()
# # Uncompressing image
# img = uncompress_image(img_compressed)

# # Displaying
# show_img_and_wait_close("img_stream", img)

# robot.close_connection()


# # from pyniryo import *

# # robot = NiryoRobot("10.10.10.10")

# # robot.calibrate_auto()
# # robot.update_tool()

# # robot.release_with_tool()
# # robot.move_pose(0.2, -0.1, 0.25, 0.0, 1.57, 0.0)
# # robot.grasp_with_tool()

# # robot.move_pose(0.2, 0.1, 0.25, 0.0, 1.57, 0.0)
# # robot.release_with_tool()

# # robot.close_connection()

# # robot.update_tool()
# # robot.open_gripper()
# # robot.close_gripper()
# # x = robot.get_current_tool_id()


# video code

from pyniryo import *

# Connecting to robot
robot = NiryoRobot("10.10.10.10")
robot.calibrate_auto()

# Getting calibration param
mtx, dist = robot.get_camera_intrinsics()
# Moving to observation pose
robot.move_pose(0.2, -0.1, 0.25, 0.0, 1.57, 0.0)

while "User do not press Escape neither Q":
    # Getting image
    img_compressed = robot.get_img_compressed()
    # Uncompressing image
    img_raw = uncompress_image(img_compressed)
    # Undistorting
    img_undistort = undistort_image(img_raw, mtx, dist)
    # Trying to find markers
    workspace_found, res_img_markers = debug_markers(img_undistort)
    # Trying to extract workspace if possible
    if workspace_found:
        img_workspace = extract_img_workspace(img_undistort, workspace_ratio=1.0)
    else:
        img_workspace = None

    if img_workspace is not None:
        resized_img_workspace = resize_img(img_workspace, height=res_img_markers.shape[0])
        res_img_markers = concat_imgs((res_img_markers, resized_img_workspace))
    # Showing images
    key = show_img("Markers", res_img_markers, wait_ms=30)
    if key in [27, ord("q")]:  # Will break loop if the user press Escape or Q
        break