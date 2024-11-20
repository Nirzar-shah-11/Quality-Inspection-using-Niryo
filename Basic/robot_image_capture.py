#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Code with image processing
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# from pyniryo import *

# # Connecting to robot
# robot = NiryoRobot("10.10.10.10")
# robot.calibrate_auto()

# # Getting calibration param
# mtx, dist = robot.get_camera_intrinsics()
# # Moving to observation pose
# robot.move_pose(0.2, -0.1, 0.25, 0.0, 1.57, 0.0)

# while "User do not press Escape neither Q":
#     # Getting image
#     img_compressed = robot.get_img_compressed()
#     # Uncompressing image
#     img_raw = uncompress_image(img_compressed)
#     # Undistorting
#     img_undistort = undistort_image(img_raw, mtx, dist)
#     # Trying to find markers
#     workspace_found, res_img_markers = debug_markers(img_undistort)
#     # Trying to extract workspace if possible
#     if workspace_found:
#         img_workspace = extract_img_workspace(img_undistort, workspace_ratio=1.0)
#     else:
#         img_workspace = None

#     if img_workspace is not None:
#         resized_img_workspace = resize_img(img_workspace, height=res_img_markers.shape[0])
#         res_img_markers = concat_imgs((res_img_markers, resized_img_workspace))
#     # Showing images
#     key = show_img("Markers", res_img_markers, wait_ms=30)
#     if key in [27, ord("q")]:  # Will break loop if the user press Escape or Q
#         break


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Code without image processing. just getting the image and flipping it 
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from pyniryo import *

# Connect to the robot
robot = NiryoRobot("10.10.10.10")
robot.calibrate_auto()

# Move the robot to the observation pose
robot.move_pose(0.2, -0.1, 0.25, 0.0, 1.57, 0.0)

while True:
    # Capture the compressed image from the robot
    img_compressed = robot.get_img_compressed()
    img_raw = uncompress_image(img_compressed)
    
    # Flip the image upside down
    img_flipped = cv2.flip(img_raw, 0)  # 0 for vertical flip

    # Display the flipped image
    key = show_img("Robot View", img_flipped, wait_ms=30)
    if key in [27, ord("q")]:  # Exit on pressing 'Esc' or 'Q'
        break
