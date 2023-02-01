# pip install --upgrade pip
# pip3 install opencv-contrib-python-headless (for docker, no GUI)
# pip3 install pyquaternion
# pip install scipy

# run in docker "colmap_opencv"
# python3 colmap_loop_linux.py

# DA FARE:
# individuazione frame statici facendo differenza sui grigi
# if a camera is not registered, it must be eliminated from the database

import subprocess
import time
import shutil
import os
import numpy as np
from pathlib import Path
import cv2
from pyquaternion import quaternion
from scipy.spatial.transform import Rotation as R
from scipy import linalg
from lib import database
from lib import static_rejection


STATIC_IMG_REJECTION_METHOD = 'radiometric' # 'radiometric' or 'root_sift'
DEBUG = False
SLEEP_TIME = 1
LOOP_CYCLES = 1000000
COLMAP_EXE_PATH = Path(r"/colmap/build/src/exe")
IMGS_FROM_SERVER = Path(r"/home/luca/Scrivania/3DOM/Github_lcmrl/Server_Connection/c++_send_images/imgs") #Path("./imgs")
MAX_N_FEATURES = "100"


### FUNCTIONS
def ExportCameras(external_cameras_path):
    lines= []
    lines.append("IMAGE_ID X Y Z NX NY NZ FOCAL_LENGTH EULER_ROTATION_MATRIX\n")
    d = {}
    k = 0
    n_images = 0
    
    with open(external_cameras_path,'r') as file :
        for line in file:
            k = k+1
            line = line[:-1]
            try:
                first_elem, waste = line.split(' ', 1)
                if first_elem == "#":
                    print(first_elem)
                elif k%2 != 0:
                    image_id, qw, qx, qy, qz, tx, ty, tz, camera_id, name = line.split(" ", 9)
                    q = np.array([float(qw), float(qx), float(qy), float(qz)])
                    t = np.array([[float(tx)],[float(ty)],[float(tz)]])
                    q_matrix = quaternion.Quaternion(q).transformation_matrix
                    q_matrix = q_matrix[0:3,0:3]
                    camera_location = np.dot(-q_matrix.transpose(),t)
                    n_images = n_images + 1
                    camera_direction = np.dot(q_matrix.transpose(),np.array([[0],[0],[1]]))#*-1
                    lines.append('{} {} {} {} {} {} {} 50 {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}\n'.format(
                                                name,
                                                camera_location[0,0],
                                                camera_location[1,0],
                                                camera_location[2,0],
                                                camera_direction[0,0],
                                                camera_direction[1,0],
                                                camera_direction[2,0],
                                                q_matrix[0,0],
                                                q_matrix[0,1],
                                                q_matrix[0,2],
                                                "0",
                                                q_matrix[1,0],
                                                q_matrix[1,1],
                                                q_matrix[1,2],
                                                "0",
                                                q_matrix[2,0],
                                                q_matrix[2,1],
                                                q_matrix[2,2],
                                                "0",
                                                "0",
                                                "0",
                                                "0",
                                                "1"
                                                ))
        
            except:
                print("Empty line")
    return lines


### MAIN STARTS HERE
CURRENT_DIR = Path(os.getcwd())
TEMP_DIR = CURRENT_DIR / "temp"
KEYFRAMES_DIR = CURRENT_DIR / "colmap_imgs"
OUT_FOLDER = CURRENT_DIR / "outs"
DATABASE = CURRENT_DIR / "outs" / "db.db"

ref_matches = []
processed_imgs = []
pointer = 0
delta = 0
ended_first_colmap_loop = False
total_imgs = "000000"

# Manage output folders
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)
    os.makedirs(TEMP_DIR / "pair")
    os.makedirs(KEYFRAMES_DIR)
    os.makedirs(OUT_FOLDER)
else:
    shutil.rmtree(TEMP_DIR)  
    shutil.rmtree(KEYFRAMES_DIR)
    shutil.rmtree(OUT_FOLDER)
    os.makedirs(TEMP_DIR)         
    os.makedirs(TEMP_DIR / "pair")
    os.makedirs(KEYFRAMES_DIR)
    os.makedirs(OUT_FOLDER)


# Main loop
for i in range (LOOP_CYCLES):
    print("LOOP: ", i)
    # Check on tie points to eliminate stationary kpts
    imgs = os.listdir(IMGS_FROM_SERVER)
    imgs = sorted(imgs, key=lambda x: int(x[6:-4])) #imgs.sort()
    print(imgs)
    newer_imgs = False
    
    # Choose if keeping the pair
    if len(imgs) < 2:
        print("[{}] len(imgs) < 2".format(i))

    elif len(imgs) >= 2:
        for c, img in enumerate(imgs):
            if img not in processed_imgs and c >= 1:
                
                img1 = imgs[pointer]
                img2 = imgs[c]
                

                print("\n[LOOP: {}]".format(i), img1, img2)
                print("pointer", pointer, pointer+1, "\n")
                
                pointer, delta, ref_matches, newer_imgs, total_imgs = static_rejection.StaticRejection(STATIC_IMG_REJECTION_METHOD, img1, img2, IMGS_FROM_SERVER, TEMP_DIR, KEYFRAMES_DIR, COLMAP_EXE_PATH, MAX_N_FEATURES, ref_matches, DEBUG, pointer, delta, newer_imgs, total_imgs)
                processed_imgs.append(img)


    kfrms = os.listdir(KEYFRAMES_DIR)
    if len(kfrms) >= 5 and newer_imgs == True:
        # Incremental reconstruction
        if DEBUG == False:
            if os.path.exists(DATABASE): subprocess.run([COLMAP_EXE_PATH / "colmap", "database_creator", "--database_path", DATABASE], stdout=subprocess.DEVNULL)
            if ended_first_colmap_loop == True:
                subprocess.run([COLMAP_EXE_PATH / "colmap", "feature_extractor", "--project_path", CURRENT_DIR / "lib" / "sift.ini"], stdout=subprocess.DEVNULL)
            elif ended_first_colmap_loop == False:
                subprocess.run([COLMAP_EXE_PATH / "colmap", "feature_extractor", "--project_path", CURRENT_DIR / "lib" / "sift_first_loop.ini"], stdout=subprocess.DEVNULL)
                ended_first_colmap_loop = True
            subprocess.run([COLMAP_EXE_PATH / "colmap", "sequential_matcher", "--database_path", DATABASE, "--SequentialMatching.overlap", "1", "--SequentialMatching.quadratic_overlap", "1"], stdout=subprocess.DEVNULL)
            subprocess.run([COLMAP_EXE_PATH / "colmap", "mapper", "--project_path", CURRENT_DIR / "lib" / "mapper.ini"], stdout=subprocess.DEVNULL)
            subprocess.run([COLMAP_EXE_PATH / "colmap", "model_converter", "--input_path", OUT_FOLDER / "0", "--output_path", OUT_FOLDER, "--output_type", "TXT"], stdout=subprocess.DEVNULL)
        
        elif DEBUG == True:
            if os.path.exists(DATABASE): subprocess.run([COLMAP_EXE_PATH / "colmap", "database_creator", "--database_path", DATABASE])
            if ended_first_colmap_loop == True:
                subprocess.run([COLMAP_EXE_PATH / "colmap", "feature_extractor", "--project_path", CURRENT_DIR / "lib" / "sift.ini"])
            elif ended_first_colmap_loop == False:
                subprocess.run([COLMAP_EXE_PATH / "colmap", "feature_extractor", "--project_path", CURRENT_DIR / "lib" / "sift_first_loop.ini"])
                ended_first_colmap_loop = True
            subprocess.run([COLMAP_EXE_PATH / "colmap", "sequential_matcher", "--database_path", DATABASE, "--SequentialMatching.overlap", "1", "--SequentialMatching.quadratic_overlap", "1"])
            subprocess.run([COLMAP_EXE_PATH / "colmap", "mapper", "--project_path", CURRENT_DIR / "lib" / "mapper.ini"])
            subprocess.run([COLMAP_EXE_PATH / "colmap", "model_converter", "--input_path", OUT_FOLDER / "0", "--output_path", OUT_FOLDER, "--output_type", "TXT"])
        
        lines = ExportCameras(OUT_FOLDER / "images.txt")
        print("EXPORTED CAMERAS POS")

        with open(OUT_FOLDER / "loc.txt", 'w') as file:
            for line in lines:
                file.write(line)

    time.sleep(SLEEP_TIME)






    
