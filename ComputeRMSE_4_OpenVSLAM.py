cameras_to_align = r'/home/luca/Github_lcmrl/COLMAP_SLAM/keyframe_trajectory_OpenVSLAM.txt'
camera_ground_truth = r'/home/luca/Github_lcmrl/COLMAP_SLAM/MAV-gt_cam0.txt'
timestamp_path = r'/home/luca/Desktop/ION2023/EuRoC_MAV/MH_01_easy/mav0/cam0/data.csv'

import os
import math
import subprocess
import numpy as np
#from sklearn.metrics import mean_squared_error



colmap_coord = {}
with open(cameras_to_align, "r") as f1:
    lines = f1.readlines()
    lines = lines[1:]
    for line in lines:
        timest, x, y, z, _ = line.strip().split(' ', 4)
        timest = int(float(timest))
        colmap_coord[str(timest)[0:15]] = (x, y, z)

timestamp_dict = {}
with open(timestamp_path, "r") as f1:
    lines = f1.readlines()
    lines = lines[1:]
    for c, line in enumerate(lines):
        timestamp, name = line.strip().split(',', 1)
        timestamp_dict[timestamp[0:15]] = c

gt_coord = {}
with open(camera_ground_truth, 'r') as f2:
    lines = f2.readlines()
    lines = lines[0:]
    for line in lines:
        timestamp, x, y, z = line.strip().split(',', 4)
        gt_coord[timestamp[0:15]] = (x, y, z)


common_keys = list(colmap_coord.keys() & gt_coord.keys())
common_keys.sort()
#print(list(colmap_coord.keys())[:10])
#print(list(gt_coord.keys())[:10]); quit()
with open("/home/luca/Desktop/ION2023/EuRoC_MAV/colmap_common_keys.txt", "w") as fx, open("/home/luca/Desktop/ION2023/EuRoC_MAV/gt_common_keys.txt", "w") as fy:
    for key in common_keys[:]: # Graphical rototranslation on first 1000 frames
        print(key)
        fx.write("{},{},{},{}\n".format(timestamp_dict[key], colmap_coord[key][0], colmap_coord[key][1], colmap_coord[key][2]))
        fy.write("{},{},{},{}\n".format(timestamp_dict[key], gt_coord[key][0], gt_coord[key][1], gt_coord[key][2]))


# Align the two clouds (cloud from COLMAP to ground_truth)
output_file = open("/home/luca/Desktop/ION2023/EuRoC_MAV/tmptmp.txt", "w")
subprocess.run(["/home/luca/Github_lcmrl/COLMAP_SLAM/AlignCC_for_linux/align", "/home/luca/Desktop/ION2023/EuRoC_MAV/colmap_common_keys.txt", "/home/luca/Desktop/ION2023/EuRoC_MAV/gt_common_keys.txt"], stdout=output_file)
output_file.close()



