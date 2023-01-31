# pip install --upgrade pip
# pip3 install opencv-contrib-python-headless (for docker, no GUI)
# pip3 install pyquaternion
# pip install scipy

# run in docker "colmap_opencv"
# python3 colmap_loop_linux.py

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

DEBUG = False
MAX_RATIO = 0.70
MIN_RATIO = 0
SLEEP_TIME = 1
LOOP_CYCLES = 1000000
COLMAP_EXE_PATH = Path(r"/colmap/build/src/exe")
IMGS_FROM_SERVER = Path(r"/home/luca/Scrivania/3DOM/Github_lcmrl/Server_Connection/c++_send_images/imgs")
MAX_N_FEATURES = "100"


### FUNCTIONS
def RootSift(img_name, desc_folder, N_kpts):

    np_kpt_path = Path("{}.kpt.npy".format(img_name))
    abs_np_kpt_path = desc_folder / np_kpt_path
    np_dsc_path = Path("{}.dsc.npy".format(img_name))
    abs_np_dsc_path = desc_folder / np_dsc_path
    kp = np.load(abs_np_kpt_path)
    desc = np.load(abs_np_dsc_path)
    kp_numb = kp.shape[0]
    
    return kp, desc, kp_numb


# Brute-Force openCV2
def BrForce(des1, des2, check, matching_distance, crossCheck_bool, matching_strategy, print_debug = True, ratio_thresh=0.8):
    if check == 'without_Lowe_ratio_test' and matching_distance=='L2':
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=crossCheck_bool)
        print(des1)
        print(des2)
        print(type(des1))
        print(type(des2))
        print(des1.shape)
        print(des2.shape)
        matches = bf.match(des1,des2)
        #matches = sorted(matches, key = lambda x: x.distance)   # Sort matches by distance.  Best come first.

        if print_debug == True :
            print('type(matches) : '), print(type(matches))
            print('shape(matches) : '), print(len(matches))
            print(matches[0]),print(matches[1]),print(matches[2]),print(matches[3])
            print(matches[0].queryIdx)
            print(matches[0].trainIdx)
            print(matches[0].distance)

        return matches
            
    elif check == 'without_Lowe_ratio_test' and matching_distance=='NORM_HAMMING':

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=crossCheck_bool)
        matches = bf.match(des1,des2)

        if print_debug == True :
            print('type(matches) : '), print(type(matches))
            print('shape(matches) : '), print(len(matches))
            print(matches[0]),print(matches[1]),print(matches[2]),print(matches[3])
            print(matches[0].queryIdx)
            print(matches[0].trainIdx)
            print(matches[0].distance)

        return matches
    
    elif check == 'Lowe_ratio_test' and matching_distance=='L2':
    
        print('check: {}'.format(check))
        print('matching_distance: {}'.format(matching_distance))
        print('matching_strategy: {}'.format(matching_strategy))
        print('ratio_thresh: {}'.format(ratio_thresh))

        bf = cv2.BFMatcher(cv2.NORM_L2, False)

        # Ratio Test
        def ratio_test(matches, ratio_thresh):
            prefiltred_matches = []
            for m,n in matches:
                #print('m={} n={}'.format(m,n))
                if m.distance < ratio_thresh * n.distance:
                    prefiltred_matches.append(m)
            return prefiltred_matches
        
        if matching_strategy == 'unidirectional':
            matches01 = bf.knnMatch(des1,des2,k=2)
            good_matches01 = ratio_test(matches01, ratio_thresh)
            return good_matches01
            
        elif matching_strategy == 'intersection':
            matches01 = bf.knnMatch(des1,des2,k=2)
            matches10 = bf.knnMatch(des2,des1,k=2)
            good_matches01 = ratio_test(matches01, ratio_thresh)
            good_matches10 = ratio_test(matches10, ratio_thresh)
            good_matches10_ = {(m.trainIdx, m.queryIdx) for m in good_matches10}
            prefiltred_matches = [m for m in good_matches01 if (m.queryIdx, m.trainIdx) in good_matches10_]
            return prefiltred_matches
            
        elif matching_strategy == 'union':
            matches01 = bf.knnMatch(des1,des2,k=2)
            matches10 = bf.knnMatch(des2,des1,k=2)
            good_matches01 = ratio_test(matches01, ratio_thresh)
            good_matches10 = ratio_test(matches10, ratio_thresh)
            good_matches10_ = {(m.trainIdx, m.queryIdx) for m in good_matches10}
            other_matches = [m for m in good_matches01 if not (m.queryIdx, m.trainIdx) in good_matches10_]
            for m in good_matches10: # added 01/10/2022 
                query = m.queryIdx; train = m.trainIdx # added 01/10/2022
                m.trainIdx = query # added 01/10/2022
                m.queryIdx = train # added 01/10/2022
            prefiltred_matches = good_matches10 + other_matches
            return prefiltred_matches
            
    elif check == 'Lowe_ratio_test' and matching_distance=='NORM_HAMMING':
    
        print('check: {}'.format(check))
        print('matching_distance: {}'.format(matching_distance))
        print('matching_strategy: {}'.format(matching_strategy))
        print('ratio_thresh: {}'.format(ratio_thresh))
        
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, False)

        # Ratio Test
        def ratio_test(matches, ratio_thresh):
            prefiltred_matches = []
            for m,n in matches:
                #print('m={} n={}'.format(m,n))
                if m.distance < ratio_thresh * n.distance:
                    prefiltred_matches.append(m)
            return prefiltred_matches
        
        if matching_strategy == 'unidirectional':
            matches01 = bf.knnMatch(des1,des2,k=2)
            good_matches01 = ratio_test(matches01, ratio_thresh)
            return good_matches01
            
        elif matching_strategy == 'intersection':
            matches01 = bf.knnMatch(des1,des2,k=2)
            matches10 = bf.knnMatch(des2,des1,k=2)
            good_matches01 = ratio_test(matches01, ratio_thresh)
            good_matches10 = ratio_test(matches10, ratio_thresh)
            good_matches10_ = {(m.trainIdx, m.queryIdx) for m in good_matches10}
            prefiltred_matches = [m for m in good_matches01 if (m.queryIdx, m.trainIdx) in good_matches10_]
            return prefiltred_matches
            
        elif matching_strategy == 'union':
            matches01 = bf.knnMatch(des1,des2,k=2)
            matches10 = bf.knnMatch(des2,des1,k=2)
            good_matches01 = ratio_test(matches01, ratio_thresh)
            good_matches10 = ratio_test(matches10, ratio_thresh)
            good_matches10_ = {(m.trainIdx, m.queryIdx) for m in good_matches10}
            other_matches = [m for m in good_matches01 if not (m.queryIdx, m.trainIdx) in good_matches10_]
            for m in good_matches10: # added 01/10/2022 
                query = m.queryIdx; train = m.trainIdx # added 01/10/2022
                m.trainIdx = query # added 01/10/2022
                m.queryIdx = train # added 01/10/2022
            prefiltred_matches = good_matches10 + other_matches
            return prefiltred_matches


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
    
    if len(imgs) < 2:
        print("[{}] len(imgs) < 2".format(i))
    elif len(imgs) >= 2:
        for c, img in enumerate(imgs):
            if img not in processed_imgs and c >= 1:
                newer_imgs = True
                shutil.rmtree(TEMP_DIR / "pair")     
                os.makedirs(TEMP_DIR / "pair")

                img1 = imgs[pointer]
                img2 = imgs[c]
                print()
                print("[LOOP: {}]".format(i), img1, img2)
                print("pointer", pointer, pointer+1)
                print()

                shutil.copy(IMGS_FROM_SERVER / "{}".format(img1), TEMP_DIR / "pair" / "{}".format(img1))
                shutil.copy(IMGS_FROM_SERVER / "{}".format(img2), TEMP_DIR / "pair" / "{}".format(img2))
                
                if DEBUG == False:
                    subprocess.run([COLMAP_EXE_PATH / "colmap", "database_creator", "--database_path", TEMP_DIR / "db.db"], stdout=subprocess.DEVNULL)
                    subprocess.run([COLMAP_EXE_PATH / "colmap", "feature_extractor", "--database_path", TEMP_DIR / "db.db", "--image_path", TEMP_DIR / "pair", "SiftExtraction.max_num_features", MAX_N_FEATURES], stdout=subprocess.DEVNULL)
                    subprocess.run(["python3", CURRENT_DIR / "lib" / "RootSIFT.py", "--Path", TEMP_DIR / "db.db", "--Output", TEMP_DIR], stdout=subprocess.DEVNULL)
                elif DEBUG == True:
                    subprocess.run([COLMAP_EXE_PATH / "colmap", "database_creator", "--database_path", TEMP_DIR / "db.db"])
                    subprocess.run([COLMAP_EXE_PATH / "colmap", "feature_extractor", "--database_path", TEMP_DIR / "db.db", "--image_path", TEMP_DIR / "pair", "SiftExtraction.max_num_features", MAX_N_FEATURES])
                    subprocess.run(["python3", CURRENT_DIR / "lib" / "RootSIFT.py", "--Path", TEMP_DIR / "db.db", "--Output", TEMP_DIR])

                os.remove(TEMP_DIR / "db.db")
                
                kp1, desc1, kp_numb1 = RootSift(img1, TEMP_DIR, 8000)
                kp2, desc2, kp_numb2 = RootSift(img2, TEMP_DIR, 8000)
                print("BrForce")
                opencv_matches = BrForce(desc1, desc2, 'Lowe_ratio_test', 'L2', True, 'intersection', print_debug = False, ratio_thresh=0.8)
                print("finish")

                matches_matrix = np.zeros((len(opencv_matches), 2))
                for l in range(0,len(opencv_matches)):
                    matches_matrix[l][0] = int(opencv_matches[l].queryIdx)
                    matches_matrix[l][1] = int(opencv_matches[l].trainIdx)
                
                if ref_matches == []:
                    ref_matches = matches_matrix
                    shutil.copy(IMGS_FROM_SERVER / "{}".format(img1), KEYFRAMES_DIR / "{}".format(img1))
                    shutil.copy(IMGS_FROM_SERVER / "{}".format(img2), KEYFRAMES_DIR / "{}".format(img2))
                    pointer += 1
                else:
                    vec_ref = ref_matches[:,1]
                    vec = matches_matrix[:,0]
                    vec_ref = vec_ref.tolist()
                    vec = vec.tolist()
                    vec_ref = [int(v) for v in vec_ref]
                    vec = [int(v) for v in vec]
                    intersection = [el for el in vec if el in vec_ref]
                    
                    control_ratio = len(intersection) / len(vec_ref)
                    print("control_ratio", control_ratio)

                    if control_ratio < MAX_RATIO and control_ratio > MIN_RATIO:
                        shutil.copy(IMGS_FROM_SERVER / "{}".format(img1), KEYFRAMES_DIR / "{}".format(img1))
                        shutil.copy(IMGS_FROM_SERVER / "{}".format(img2), KEYFRAMES_DIR / "{}".format(img2))
                        print("\n.. added img\n")
                        ref_matches = matches_matrix
                        pointer += 1 + delta
                        delta = 0
                    else:
                        delta += 1
                        print("\n.. NO\n")
                        
                processed_imgs.append(img)

    kfrms = os.listdir(KEYFRAMES_DIR)
    if len(kfrms) >= 3 and newer_imgs == True:
        # Incremental reconstruction
        if DEBUG == False:
            subprocess.run([COLMAP_EXE_PATH / "colmap", "database_creator", "--database_path", DATABASE], stdout=subprocess.DEVNULL)
            if ended_first_colmap_loop == True:
                subprocess.run([COLMAP_EXE_PATH / "colmap", "feature_extractor", "--project_path", CURRENT_DIR / "lib" / "sift.ini"], stdout=subprocess.DEVNULL)
            elif ended_first_colmap_loop == False:
                subprocess.run([COLMAP_EXE_PATH / "colmap", "feature_extractor", "--project_path", CURRENT_DIR / "lib" / "sift_first_loop.ini"], stdout=subprocess.DEVNULL)
                ended_first_colmap_loop = True
            subprocess.run([COLMAP_EXE_PATH / "colmap", "sequential_matcher", "--database_path", DATABASE, "--SequentialMatching.overlap", "2"], stdout=subprocess.DEVNULL)
            subprocess.run([COLMAP_EXE_PATH / "colmap", "mapper", "--project_path", CURRENT_DIR / "lib" / "mapper.ini"], stdout=subprocess.DEVNULL)
            subprocess.run([COLMAP_EXE_PATH / "colmap", "model_converter", "--input_path", OUT_FOLDER / "0", "--output_path", OUT_FOLDER, "--output_type", "TXT"], stdout=subprocess.DEVNULL)
        elif DEBUG == True:
            subprocess.run([COLMAP_EXE_PATH / "colmap", "database_creator", "--database_path", DATABASE])
            if ended_first_colmap_loop == True:
                subprocess.run([COLMAP_EXE_PATH / "colmap", "feature_extractor", "--project_path", CURRENT_DIR / "lib" / "sift.ini"])
            elif ended_first_colmap_loop == False:
                subprocess.run([COLMAP_EXE_PATH / "colmap", "feature_extractor", "--project_path", CURRENT_DIR / "lib" / "sift_first_loop.ini"])
                ended_first_colmap_loop = True
            subprocess.run([COLMAP_EXE_PATH / "colmap", "sequential_matcher", "--database_path", DATABASE, "--SequentialMatching.overlap", "2"])
            subprocess.run([COLMAP_EXE_PATH / "colmap", "mapper", "--project_path", CURRENT_DIR / "lib" / "mapper.ini"])
            subprocess.run([COLMAP_EXE_PATH / "colmap", "model_converter", "--input_path", OUT_FOLDER / "0", "--output_path", OUT_FOLDER, "--output_type", "TXT"])
        
        lines = ExportCameras(OUT_FOLDER / "images.txt")
        print("EXPORTED CAMERAS POS")

        with open(OUT_FOLDER / "loc.txt", 'w') as file:
            for line in lines:
                file.write(line)

    time.sleep(SLEEP_TIME)






    
