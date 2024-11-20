from pyniryo import *
import numpy as np
import time
import math
robot = NiryoRobot("10.10.10.10")
robot.calibrate_auto()
robot.update_tool()
robot.release_with_tool()
time.sleep(1)
robot.grasp_with_tool()
robot.move_joints(0.5,-0.3,0.1,0.57,0,0)
time.sleep(1)
robot.release_with_tool()
robot.close_connection()

def convert_data(data):
#converts centimeter to meter pos_cm
    pos_cm = np.array([data[0],data[1],data[2]])
    pos_m = np.array(pos_cm/100)
    pos_m = np.array(pos_m/2)

    end_d =np.array ([data[3], data[4],data[5]])
    end_r = np.array(end_d) * ((math.pi) /180)
    end_r = np. around (end_r,2)
#sort data into an array 
    j1 = pos_m[0]
    j2 = pos_m[1]
    j3 = pos_m[2]
    j4 = end_r[1]    
    j5 = end_r[0]
    j6 = end_r[1]
    convert = (j1,j2,j3,j4,j5,j6)
    return(convert)
def main():

    start_up = ([0,20,30,0,90,0])
    start_down = ([0,20,9,0,90,0])
    end_up = ([20,0,30,0,90,90])
    end_down = ([20,0,9,0,90,90])

    pos = (convert_data(start_up))
    print("pos",pos)
    robot.move_pose(pos)
    time.sleep(0.5)

    pos = (convert_data(start_down))
    print("pos: ",pos) 
    robot.move_pose(pos)
    time.sleep(0.5) 

    robot.grasp_with_tool()

    pos = (convert_data(start_up))
    print("pos",pos)
    robot.move_pose(pos)
    time.sleep(0.5)

    pos = (convert_data(end_up))
    print("pos: ",pos) 
    robot.move_pose(pos)
    time.sleep(0.5) 

    robot.release_with_tool()

    pos = (convert_data(end_down))
    print("pos: ",pos) 
    robot.move_pose(pos)
    time.sleep(0.5) 

if __name__ == "__main__":
    main()
    robot.close_connection()