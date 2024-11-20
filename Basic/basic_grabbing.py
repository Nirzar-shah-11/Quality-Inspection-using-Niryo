from pyniryo import *
robot = NiryoRobot("10.10.10.10")
robot.calibrate_auto()
robot.calibrate_auto()
joint_angles = robot.get_joints()
print("Current joint angles:", joint_angles)
# robot.update_tool()
# robot.move_joints(0.5,0.3,0.1,0.57,0.1,0)
robot.move_joints(0.10124025646141721, 0.163090991734332, -1.3112160299761095, 0.1504227708025856, 0.5782181034430938, -0.30670350398733515)
# robot.grasp_with_tool()
# robot.release_with_tool()
# robot.close_connection()

# robot.update_tool()