import numpy as np
from pathlib import Path
from pyquaternion import quaternion

#t_bs = np.array([[7.48903e-02, -1.84772e-02, -1.20209e-01]]).T # LEICA
#t_bs = np.array([[-0.0216401454975, -0.064676986768, 0.00981073058949]]).T # cam0
t_bs = np.array([[-0.0198435579556, 0.0453689425024, 0.00786212447038]]).T # cam1

path_to_Trb_csv = Path(r"/home/luca/Desktop/ION2023/EuRoC_MAV/MH_01_easy/mav0/state_groundtruth_estimate0/data.csv")

with open(path_to_Trb_csv, 'r') as Trb_csv, open('./MAV-gt.txt', 'w') as out:
    lines = Trb_csv.readlines()
    for line in lines[1:]:
        timestamp, p_x, p_y, p_z, q_w, q_x, q_y, q_z, _ = line.split(',', 8)
        quat = quaternion.Quaternion(np.array([q_w, q_x, q_y, q_z]))
        rotation_matrix = quat.inverse.rotation_matrix#quat.inverse.rotation_matrix
        translation = np.dot(rotation_matrix, t_bs)
        x = float(p_x) + translation[0,0]
        y = float(p_y) + translation[1,0]
        z = float(p_z) + translation[2,0]
        out.write('{},{},{},{}\n'.format(timestamp, x, y, z))