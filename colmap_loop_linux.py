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
from lib import export_cameras


STATIC_IMG_REJECTION_METHOD = 'radiometric' # 'radiometric' or 'root_sift'
DEBUG = False
SLEEP_TIME = 1/5
LOOP_CYCLES = 1000000
COLMAP_EXE_PATH = Path(r"/colmap/build/src/exe")
IMGS_FROM_SERVER = Path(r"/home/luca/Scrivania/3DOM/Github_lcmrl/Server_Connection/c++_send_images/imgs") #Path("./imgs")
MAX_N_FEATURES = "100"


### FUNCTIONS



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
    if len(kfrms) >= 10 and newer_imgs == True:
        # Incremental reconstruction
        if DEBUG == False:
            if os.path.exists(DATABASE): subprocess.run([COLMAP_EXE_PATH / "colmap", "database_creator", "--database_path", DATABASE], stdout=subprocess.DEVNULL)
            if ended_first_colmap_loop == True:
                subprocess.run([COLMAP_EXE_PATH / "colmap", "feature_extractor", "--project_path", CURRENT_DIR / "lib" / "sift.ini"], stdout=subprocess.DEVNULL)
            elif ended_first_colmap_loop == False:
                subprocess.run([COLMAP_EXE_PATH / "colmap", "feature_extractor", "--project_path", CURRENT_DIR / "lib" / "sift_first_loop.ini"], stdout=subprocess.DEVNULL)
                ended_first_colmap_loop = True
            subprocess.run([COLMAP_EXE_PATH / "colmap", "sequential_matcher", "--database_path", DATABASE, "--SequentialMatching.overlap", "10", "--SequentialMatching.quadratic_overlap", "1"], stdout=subprocess.DEVNULL)
            if os.path.exists(OUT_FOLDER / "0"):
                subprocess.run([COLMAP_EXE_PATH / "colmap", "mapper", "--project_path", CURRENT_DIR / "lib" / "mapper.ini"], stdout=subprocess.DEVNULL)
                subprocess.run([COLMAP_EXE_PATH / "colmap", "model_converter", "--input_path", OUT_FOLDER, "--output_path", OUT_FOLDER, "--output_type", "TXT"], stdout=subprocess.DEVNULL)
            else:
                subprocess.run([COLMAP_EXE_PATH / "colmap", "mapper", "--project_path", CURRENT_DIR / "lib" / "mapper_first_loop.ini"], stdout=subprocess.DEVNULL)
                subprocess.run([COLMAP_EXE_PATH / "colmap", "model_converter", "--input_path", OUT_FOLDER / "0", "--output_path", OUT_FOLDER, "--output_type", "TXT"], stdout=subprocess.DEVNULL)
        
        elif DEBUG == True:
            if os.path.exists(DATABASE): subprocess.run([COLMAP_EXE_PATH / "colmap", "database_creator", "--database_path", DATABASE])
            if ended_first_colmap_loop == True:
                subprocess.run([COLMAP_EXE_PATH / "colmap", "feature_extractor", "--project_path", CURRENT_DIR / "lib" / "sift.ini"])
            elif ended_first_colmap_loop == False:
                subprocess.run([COLMAP_EXE_PATH / "colmap", "feature_extractor", "--project_path", CURRENT_DIR / "lib" / "sift_first_loop.ini"])
                ended_first_colmap_loop = True
            subprocess.run([COLMAP_EXE_PATH / "colmap", "sequential_matcher", "--database_path", DATABASE, "--SequentialMatching.overlap", "10", "--SequentialMatching.quadratic_overlap", "1"])
            if os.path.exists(OUT_FOLDER / "0"):
                subprocess.run([COLMAP_EXE_PATH / "colmap", "mapper", "--project_path", CURRENT_DIR / "lib" / "mapper.ini"])
                subprocess.run([COLMAP_EXE_PATH / "colmap", "model_converter", "--input_path", OUT_FOLDER, "--output_path", OUT_FOLDER, "--output_type", "TXT"])
            else:
                subprocess.run([COLMAP_EXE_PATH / "colmap", "mapper", "--project_path", CURRENT_DIR / "lib" / "mapper_first_loop.ini"])
                subprocess.run([COLMAP_EXE_PATH / "colmap", "model_converter", "--input_path", OUT_FOLDER / "0", "--output_path", OUT_FOLDER, "--output_type", "TXT"])
        
        lines = export_cameras.ExportCameras(OUT_FOLDER / "images.txt")
        print("EXPORTED CAMERAS POS")

        with open(OUT_FOLDER / "loc.txt", 'w') as file:
            for line in lines:
                file.write(line)

    time.sleep(SLEEP_TIME)






    
