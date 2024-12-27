from pyniryo import *

# Connect to the robot
robot = NiryoRobot("10.10.10.10")  # Replace with the robot's IP address

# Start robot calibration
print("Starting robot calibration...")
robot.calibrate_auto()
print("Calibration complete.")

# Define the workspace name
workspace_name = "QualityInspection"

# Record the four corners of the workspace
print("Please manually move the robot to the bottom-left corner of the workspace and confirm.")
input("Press Enter once the robot is positioned correctly...")
bottom_left_pose = robot.get_pose()
print("Bottom-left corner recorded:", bottom_left_pose)

print("Please manually move the robot to the bottom-right corner of the workspace and confirm.")
input("Press Enter once the robot is positioned correctly...")
bottom_right_pose = robot.get_pose()
print("Bottom-right corner recorded:", bottom_right_pose)

print("Please manually move the robot to the top-left corner of the workspace and confirm.")
input("Press Enter once the robot is positioned correctly...")
top_left_pose = robot.get_pose()
print("Top-left corner recorded:", top_left_pose)

print("Please manually move the robot to the top-right corner of the workspace and confirm.")
input("Press Enter once the robot is positioned correctly...")
top_right_pose = robot.get_pose()
print("Top-right corner recorded:", top_right_pose)

# Save workspace details manually
workspace_bounds = {
    "bottom_left": bottom_left_pose,
    "bottom_right": bottom_right_pose,
    "top_left": top_left_pose,
    "top_right": top_right_pose,
}

print(f"Workspace '{workspace_name}' recorded successfully with the following bounds:")
for corner, pose in workspace_bounds.items():
    print(f"{corner}: {pose}")

# Optional: Save workspace bounds to a file for reuse
with open(f"{workspace_name}_workspace_bounds.txt", "w") as file:
    for corner, pose in workspace_bounds.items():
        file.write(f"{corner}: {pose}\n")

# Disconnect from the robot
robot.close_connection()
