from pyniryo import *
import numpy as np
import math
import time

work_space = "QualityInspection"
robot = NiryoRobot("10.10.10.10")

def convert_data(data):
    pos_cm = np.array([data[0],data[1],data[2]])
    pos_m = np.array(pos_cm/100)
    pos_m = np.around(pos_m,2)
    end_d = np.array([data[3],data[4],data[5]])
    end_r = np.array(end_d) * (math.pi / 180)
    end_r = np.around(end_r,2)

    #sort data
    j1 = pos_m[0]
    j2 = pos_m[1]
    j3 = pos_m[2]
    j4 = end_r[0]
    j5 = end_r[1]
    j6 = end_r[2]

    convert = (j1,j2,j3,j4,j5,j6)
    return (convert)

#calibrate robot 
robot.calibrate_auto()
#updating tool 
robot.update_tool()

# Define robot position 
home = ([25,0,30,0,90,0]) #observation
obs = ([25,0,40,0,90,0]) #observation
place_up = ([0,17,25,0,90,90])
place_down = ([0,17,9,0,90,90])

# Initialise variables
max_catch_count = 1
catch_count = 0

while catch_count < max_catch_count:
    robot.release_with_tool()
    # Moving to observation pose]
    pos = (convert_data(obs))
    robot.move_pose(pos)

    #trying to get an object via Vision pick
    obj_found, shape_ret, color_ret = robot.vision_pick(work_space,
                                                        height_offset=0.05,
                                                        shape=ObjectShape.ANY,
                                                        color=ObjectColor.ANY)
    if not obj_found:
        robot.wait(0.1)
        continue
    else:
        #travel to drop pos 
        pos = (convert_data(place_up))
        robot.move_pose(pos)
        time.sleep(1)

        #drop the object
        pos = (convert_data(place_down))
        robot.move_pose(pos)
        time.sleep(1)
        robot.release_with_tool()

    catch_count += 1
    print("Object Count:"+ str(catch_count))

robot.close_connection()




   