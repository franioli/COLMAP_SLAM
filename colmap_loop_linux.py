# pip install --upgrade pip
# pip3 install opencv-contrib-python-headless (for docker, no GUI)
# pip3 install pyquaternion
# pip install scipy

# run in docker "colmap_opencv"
# python3 colmap_loop_linux.py

# DA FARE:
# Server-Client c'è un problema con l'ordine dei file trovati nella cartella, vanno ordinati secondo un criterio
# migliorare plot 3d
# VELOCIZZARE TUTTO PER PROCESSARE PIU' FRAMES AL SECONDO

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


DEBUG = False
STATIC_IMG_REJECTION_METHOD = 'root_sift' # 'radiometric' or 'root_sift'
SLEEP_TIME = 1/10
LOOP_CYCLES = 1000000
COLMAP_EXE_PATH = Path(r"/colmap/build/src/exe")
IMGS_FROM_SERVER = Path(r"/home/luca/Scrivania/3DOM/Github_lcmrl/Server_Connection/c++_send_images/imgs") #Path("./imgs")
MAX_N_FEATURES = "100"
INITIAL_SEQUENTIAL_OVERLAP = 2
SEQUENTIAL_OVERLAP = 2


### FUNCTIONS
def Id2name(id):
    if id < 10:
        img_name = "00000{}.jpg".format(id)
    elif id < 100:
        img_name = "0000{}.jpg".format(id)
    elif id < 1000:
        img_name = "000{}.jpg".format(id)
    elif id < 10000:
        img_name = "00{}.jpg".format(id)
    elif id < 100000:
        img_name = "0{}.jpg".format(id)
    elif id < 1000000:
        img_name = "{}.jpg".format(id)
    return img_name


### MAIN STARTS HERE
CURRENT_DIR = Path(os.getcwd())
TEMP_DIR = CURRENT_DIR / "temp"
KEYFRAMES_DIR = CURRENT_DIR / "colmap_imgs"
OUT_FOLDER = CURRENT_DIR / "outs"
DATABASE = CURRENT_DIR / "outs" / "db.db"

img_dict = {}
ref_matches = []
processed_imgs = []
img_batch = []
img_batch_n = []
oriented_imgs_batch = []
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
    print("(pointer, delta) = ({}, {})".format(pointer, delta))
    # Check on tie points to eliminate stationary kpts
    imgs = os.listdir(IMGS_FROM_SERVER)
    imgs = sorted(imgs, key=lambda x: int(x[6:-4])) #imgs.sort()
    #print(imgs)
    newer_imgs = False
    
    # Choose if keeping the pair
    if len(imgs) < 2:
        print("[{}] len(imgs) < 2".format(i))

    elif len(imgs) >= 2:
        for c, img in enumerate(imgs):
            # Decide if new images are valid to be added to the sequential matching
            if img not in processed_imgs and c >= 1 and c != pointer and c > pointer+delta:
                img1 = imgs[pointer]
                img2 = imgs[c]
                print("\n[LOOP: {}]".format(i), img1, img2)
                ref_matches, newer_imgs, total_imgs, img_dict, img_batch, pointer = static_rejection.StaticRejection(STATIC_IMG_REJECTION_METHOD, img1, img2, IMGS_FROM_SERVER, CURRENT_DIR, KEYFRAMES_DIR, COLMAP_EXE_PATH, MAX_N_FEATURES, ref_matches, DEBUG, newer_imgs, total_imgs, img_dict, img_batch, pointer) # pointer, delta, 
                processed_imgs.append(img)
                


    kfrms = os.listdir(KEYFRAMES_DIR)
    if len(kfrms) >= 3 and newer_imgs == True: # 3 is mandatory or the pointer will not updated untill min of len(kfrms) is reached

        # Incremental reconstruction
        if os.path.exists(DATABASE): subprocess.run([COLMAP_EXE_PATH / "colmap", "database_creator", "--database_path", DATABASE], stdout=subprocess.DEVNULL)
        if ended_first_colmap_loop == True:
            subprocess.run([COLMAP_EXE_PATH / "colmap", "feature_extractor", "--project_path", CURRENT_DIR / "lib" / "sift.ini"], stdout=subprocess.DEVNULL)
        elif ended_first_colmap_loop == False:
            subprocess.run([COLMAP_EXE_PATH / "colmap", "feature_extractor", "--project_path", CURRENT_DIR / "lib" / "sift_first_loop.ini"], stdout=subprocess.DEVNULL)
            ended_first_colmap_loop = True
        subprocess.run([COLMAP_EXE_PATH / "colmap", "sequential_matcher", "--database_path", DATABASE, "--SequentialMatching.overlap", "{}".format(SEQUENTIAL_OVERLAP), "--SequentialMatching.quadratic_overlap", "1"], stdout=subprocess.DEVNULL)
        if os.path.exists(OUT_FOLDER / "0"):
            subprocess.run([COLMAP_EXE_PATH / "colmap", "mapper", "--project_path", CURRENT_DIR / "lib" / "mapper.ini"], stdout=subprocess.DEVNULL)
            subprocess.run([COLMAP_EXE_PATH / "colmap", "model_converter", "--input_path", OUT_FOLDER, "--output_path", OUT_FOLDER, "--output_type", "TXT"], stdout=subprocess.DEVNULL)
        else:
            subprocess.run([COLMAP_EXE_PATH / "colmap", "mapper", "--project_path", CURRENT_DIR / "lib" / "mapper_first_loop.ini"], stdout=subprocess.DEVNULL)
            subprocess.run([COLMAP_EXE_PATH / "colmap", "model_converter", "--input_path", OUT_FOLDER / "0", "--output_path", OUT_FOLDER, "--output_type", "TXT"], stdout=subprocess.DEVNULL)
        
        lines, oriented_dict = export_cameras.ExportCameras(OUT_FOLDER / "images.txt")

        with open(OUT_FOLDER / "loc.txt", 'w') as file:
            for line in lines:
                file.write(line)
        print("EXPORTED CAMERAS POS")

        for im_input_format in img_batch:
            im_zero_format = img_dict[im_input_format]
            img_batch_n.append(int(im_zero_format[:-4]))
            if int(im_zero_format[:-4]) in list(oriented_dict.keys()):
                oriented_imgs_batch.append(int(im_zero_format[:-4]))

        #if len(oriented_imgs_batch) != 0:
        # Define new reference img (pointer)
        last_img_n = max(list(oriented_dict.keys())) #max(oriented_imgs_batch)
        max_img_n = max(img_batch_n)
        img_name = Id2name(last_img_n)
        inverted_img_dict = {v: k for k, v in img_dict.items()}
        for c, el in enumerate(imgs):
            print(c,el)
            if el == inverted_img_dict[img_name]:
                pointer = c
        #pointer = imgs.index(inverted_img_dict[img_name])
        delta = max_img_n - last_img_n
        img_batch = []
        oriented_imgs_batch = []

        if delta != 0:
            SEQUENTIAL_OVERLAP = INITIAL_SEQUENTIAL_OVERLAP + 2*delta
        else:
            SEQUENTIAL_OVERLAP = INITIAL_SEQUENTIAL_OVERLAP


        print(img_dict)
        print()
        print(imgs)
        print("last_img_n, img_name, inverted_img_dict[img_name], pointer, delta")
        print(last_img_n, img_name, inverted_img_dict[img_name], pointer, delta)


    time.sleep(SLEEP_TIME)






    
