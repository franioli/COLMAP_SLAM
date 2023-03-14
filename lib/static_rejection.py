import os
import shutil
import subprocess
from PIL import Image, ImageOps
import numpy as np

from lib import database
from lib import BruteForce

MAX_RATIO = 0.90 #0.60
MIN_RATIO = 0

# PARAM FOR THE RADIOMETRIC APPROACH
# Try to normalize respect mean and std to reject static frames
RESIZE_SCALE_FACTOR = 1 # It can be usefull for reduce computation time
INNOVATION_THRESH = 0.001 # 1.5

def RootSift(img_name, desc_folder, N_kpts):

    np_kpt_path = Path("{}.kpt.npy".format(img_name))
    abs_np_kpt_path = desc_folder / np_kpt_path
    np_dsc_path = Path("{}.dsc.npy".format(img_name))
    abs_np_dsc_path = desc_folder / np_dsc_path
    kp = np.load(abs_np_kpt_path)
    desc = np.load(abs_np_dsc_path)
    kp_numb = kp.shape[0]
    
    return kp, desc, kp_numb


def NextImg(last_img):
    if last_img+1 < 10:
        next_img = "00000{}".format(last_img+1)
    elif last_img+1 < 100:
        next_img = "0000{}".format(last_img+1)
    elif last_img+1 < 1000:
        next_img = "000{}".format(last_img+1)
    elif last_img+1 < 10000:
        next_img = "00{}".format(last_img+1)
    elif last_img+1 < 100000:
        next_img = "0{}".format(last_img+1)
    elif last_img+1 < 1000000:
        next_img = "{}".format(last_img+1)
    return next_img


def StaticRejection(STATIC_IMG_REJECTION_METHOD, img1, img2, IMGS_FROM_SERVER, CURRENT_DIR, KEYFRAMES_DIR, COLMAP_EXE_PATH, MAX_N_FEATURES, ref_matches, DEBUG, newer_imgs, last_img, img_dict, img_batch, pointer, colmap_exe):
    # ROOTSIFT APPROACH
    if STATIC_IMG_REJECTION_METHOD == 'root_sift':
        TEMP_DIR = CURRENT_DIR / "temp"
        shutil.rmtree(TEMP_DIR / "pair")
        os.makedirs(TEMP_DIR / "pair")
        shutil.copy(IMGS_FROM_SERVER / "{}".format(img1), TEMP_DIR / "pair" / "{}".format(img1))
        shutil.copy(IMGS_FROM_SERVER / "{}".format(img2), TEMP_DIR / "pair" / "{}".format(img2))

        subprocess.run([COLMAP_EXE_PATH / f"{colmap_exe}", "database_creator", "--database_path", TEMP_DIR / "db.db"], stdout=subprocess.DEVNULL)
        subprocess.run([COLMAP_EXE_PATH / f"{colmap_exe}", "feature_extractor", "--database_path", TEMP_DIR / "db.db", "--image_path", TEMP_DIR / "pair", "SiftExtraction.max_num_features", str(MAX_N_FEATURES)], stdout=subprocess.DEVNULL)
        #subprocess.run(["python3", CURRENT_DIR / "lib" / "RootSIFT.py", "--Path", TEMP_DIR / "db.db", "--Output", TEMP_DIR], stdout=subprocess.DEVNULL)
        subprocess.run([COLMAP_EXE_PATH / f"{colmap_exe}", "sequential_matcher", "--database_path", TEMP_DIR / "db.db", "--SequentialMatching.overlap", "1"], stdout=subprocess.DEVNULL)
        #subprocess.run([COLMAP_EXE_PATH / f"{colmap_exe}", "mapper", "--project_path", CURRENT_DIR / "lib" / "mapper_for_static_rejection.ini"], stdout=subprocess.DEVNULL)

        #kp1, desc1, kp_numb1 = RootSift(img1, TEMP_DIR, 8000)
        #kp2, desc2, kp_numb2 = RootSift(img2, TEMP_DIR, 8000)
        #opencv_matches = BrForce(desc1, desc2, 'Lowe_ratio_test', 'L2', True, 'intersection', print_debug = False, ratio_thresh=0.8
        #matches_matrix = np.zeros((len(opencv_matches), 2))
        #for l in range(0,len(opencv_matches)):
        #    matches_matrix[l][0] = int(opencv_matches[l].queryIdx)
        #    matches_matrix[l][1] = int(opencv_matches[l].trainIdx

        db_p = TEMP_DIR / "db.db"
        matches = database.dbReturnMatches(db_p.as_posix(), 15)
        os.remove(TEMP_DIR / "db.db")

        if len(matches.keys()) != 0:
            key = list(matches.keys())[0]
            matches_matrix = matches[key]

            if ref_matches == []:
                ref_matches = matches_matrix
                shutil.copy(IMGS_FROM_SERVER / "{}".format(img1), KEYFRAMES_DIR / "{}.jpg".format(NextImg(int(last_img))))
                shutil.copy(IMGS_FROM_SERVER / "{}".format(img2), KEYFRAMES_DIR / "{}.jpg".format(NextImg(int(last_img)+1)))
                img_dict["{}".format(img1)] = "{}.jpg".format(NextImg(int(last_img)))
                img_dict["{}".format(img2)] = "{}.jpg".format(NextImg(int(last_img)+1))
                pointer += 1
                return ref_matches, newer_imgs, NextImg(int(last_img)+1), img_dict, img_batch, pointer # pointer, delta, 
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

                if control_ratio < MAX_RATIO and control_ratio > MIN_RATIO: # and os.path.exists(TEMP_DIR / "0"):
                    #shutil.copy(IMGS_FROM_SERVER / "{}".format(img1), KEYFRAMES_DIR / "{}".format(img1))
                    shutil.copy(IMGS_FROM_SERVER / "{}".format(img2), KEYFRAMES_DIR / "{}.jpg".format(NextImg(int(last_img))))
                    img_dict["{}".format(img2)] = "{}.jpg".format(NextImg(int(last_img)))
                    print("\n.. added img\n")
                    ref_matches = matches_matrix
                    pointer += 1 #+ delta
                    #delta = 0
                    newer_imgs = True
                    img_batch.append(img2)
                    return ref_matches, newer_imgs, NextImg(int(last_img)), img_dict, img_batch, pointer # pointer, delta, 

                else:
                    #delta += 1
                    print("\n.. NO\n")
                    return ref_matches, newer_imgs, last_img, img_dict, img_batch, pointer # pointer, delta, 
                
        elif len(matches.keys()) == 0:
            #delta += 1
            print("\n.. NO .. len(matches.keys()) == 0\n")
            return ref_matches, newer_imgs, last_img, img_dict, img_batch, pointer # pointer, delta, 

    # RADIOMETRIC APPROACH
    elif STATIC_IMG_REJECTION_METHOD == 'radiometric':
        # 'Try' is necessary because main loop looks for new images and the last one can be incomplete because
        # it is copied from other folders, and the procedure can be unfineshed
        try:
            im1 = Image.open(IMGS_FROM_SERVER / img1)
            im2 = Image.open(IMGS_FROM_SERVER / img2)
            im1.resize((round(im1.size[0]*RESIZE_SCALE_FACTOR), round(im1.size[1]*RESIZE_SCALE_FACTOR)))
            im2.resize((round(im2.size[0]*RESIZE_SCALE_FACTOR), round(im2.size[1]*RESIZE_SCALE_FACTOR)))
            im1_gray = ImageOps.grayscale(im1)
            im2_gray = ImageOps.grayscale(im2)

            # Normalization
            im1_array = np.array(im1_gray)
            im1_array = (im1_array - np.min(im1_array))/np.max(im1_array)
            im2_array = np.array(im2_gray)
            im2_array = (im2_array - np.min(im2_array))/np.max(im2_array)

            mean1 = np.mean(im1_array)
            mean2 = np.mean(im2_array)

            #innovation = np.sum(((im1_array - np.mean(im1_array)) * (im2_array - np.mean(im2_array))) / (np.std(im1_array) * np.std(im2_array)))
            #ref = np.sum(((im1_array - np.mean(im1_array)) * (im1_array - np.mean(im1_array))) / (np.std(im1_array) * np.std(im1_array)))
            #innovation = innovation/ref

            innovation = np.absolute(mean2 - mean1)

            if innovation > INNOVATION_THRESH:
                if ref_matches == []:
                    ref_matches = ["-"] # It is used for compatibilities with frame rejection approches that needs matches matrix
                    shutil.copy(IMGS_FROM_SERVER / "{}".format(img1), KEYFRAMES_DIR / "{}.jpg".format(NextImg(int(last_img))))
                    shutil.copy(IMGS_FROM_SERVER / "{}".format(img2), KEYFRAMES_DIR / "{}.jpg".format(NextImg(int(last_img)+1)))
                    img_dict["{}".format(img1)] = "{}.jpg".format(NextImg(int(last_img)))
                    img_dict["{}".format(img2)] = "{}.jpg".format(NextImg(int(last_img)+1))
                    pointer += 1
                    return ref_matches, newer_imgs, NextImg(int(last_img)+1), img_dict, img_batch, pointer

                elif ref_matches == ["-"]:
                    shutil.copy(IMGS_FROM_SERVER / "{}".format(img2), KEYFRAMES_DIR / "{}.jpg".format(NextImg(int(last_img))))
                    img_dict["{}".format(img2)] = "{}.jpg".format(NextImg(int(last_img)))
                    pointer += 1
                    newer_imgs = True
                    img_batch.append(img2)
                    return ref_matches, newer_imgs, NextImg(int(last_img)), img_dict, img_batch, pointer

            else:
                print("!! Frame rejeccted. innovation < INNOVATION_THRESH !!", end='\r')
                return ref_matches, newer_imgs, last_img, img_dict, img_batch, pointer

        except:
            print("!! Frame truncated !!")
            return ref_matches, newer_imgs, last_img, img_dict, img_batch, pointer
    
    else:
        print("Choose 'radiometric' or 'root_sift' as STATIC_IMG_REJECTION_METHOD")
        quit()
