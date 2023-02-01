import os
import shutil
import subprocess
from PIL import Image, ImageOps
import numpy as np

from lib import database

MAX_RATIO = 0.60
MIN_RATIO = 0

####
#### RootSift
####

def RootSift(img_name, desc_folder, N_kpts):

    np_kpt_path = Path("{}.kpt.npy".format(img_name))
    abs_np_kpt_path = desc_folder / np_kpt_path
    np_dsc_path = Path("{}.dsc.npy".format(img_name))
    abs_np_dsc_path = desc_folder / np_dsc_path
    kp = np.load(abs_np_kpt_path)
    desc = np.load(abs_np_dsc_path)
    kp_numb = kp.shape[0]
    
    return kp, desc, kp_numb


####
#### Brute-Force openCV2
####

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


####
#### StaticRejection
####

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

def StaticRejection(STATIC_IMG_REJECTION_METHOD, img1, img2, IMGS_FROM_SERVER, TEMP_DIR, KEYFRAMES_DIR, COLMAP_EXE_PATH, MAX_N_FEATURES, ref_matches, DEBUG, pointer, delta, newer_imgs, last_img):
    if STATIC_IMG_REJECTION_METHOD == 'root_sift':

        shutil.rmtree(TEMP_DIR / "pair")     
        os.makedirs(TEMP_DIR / "pair")
        shutil.copy(IMGS_FROM_SERVER / "{}".format(img1), TEMP_DIR / "pair" / "{}".format(img1))
        shutil.copy(IMGS_FROM_SERVER / "{}".format(img2), TEMP_DIR / "pair" / "{}".format(img2))

        if DEBUG == False:
            subprocess.run([COLMAP_EXE_PATH / "colmap", "database_creator", "--database_path", TEMP_DIR / "db.db"], stdout=subprocess.DEVNULL)
            subprocess.run([COLMAP_EXE_PATH / "colmap", "feature_extractor", "--database_path", TEMP_DIR / "db.db", "--image_path", TEMP_DIR / "pair", "SiftExtraction.max_num_features", MAX_N_FEATURES], stdout=subprocess.DEVNULL)
            #subprocess.run(["python3", CURRENT_DIR / "lib" / "RootSIFT.py", "--Path", TEMP_DIR / "db.db", "--Output", TEMP_DIR], stdout=subprocess.DEVNULL)
            subprocess.run([COLMAP_EXE_PATH / "colmap", "sequential_matcher", "--database_path", TEMP_DIR / "db.db", "--SequentialMatching.overlap", "1"], stdout=subprocess.DEVNULL)
        elif DEBUG == True:
            subprocess.run([COLMAP_EXE_PATH / "colmap", "database_creator", "--database_path", TEMP_DIR / "db.db"])
            subprocess.run([COLMAP_EXE_PATH / "colmap", "feature_extractor", "--database_path", TEMP_DIR / "db.db", "--image_path", TEMP_DIR / "pair", "SiftExtraction.max_num_features", MAX_N_FEATURES])
            #subprocess.run(["python3", CURRENT_DIR / "lib" / "RootSIFT.py", "--Path", TEMP_DIR / "db.db", "--Output", TEMP_DIR])
            subprocess.run([COLMAP_EXE_PATH / "colmap", "sequential_matcher", "--database_path", TEMP_DIR / "db.db", "--SequentialMatching.overlap", "1"])
        
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
                pointer += 1
                return pointer, delta, ref_matches, newer_imgs, NextImg(int(last_img)+1)
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
                    #shutil.copy(IMGS_FROM_SERVER / "{}".format(img1), KEYFRAMES_DIR / "{}".format(img1))
                    shutil.copy(IMGS_FROM_SERVER / "{}".format(img2), KEYFRAMES_DIR / "{}.jpg".format(NextImg(int(last_img))))
                    print("\n.. added img\n")
                    ref_matches = matches_matrix
                    pointer += 1 + delta
                    delta = 0
                    newer_imgs = True
                    return pointer, delta, ref_matches, newer_imgs, NextImg(int(last_img))

                else:
                    delta += 1
                    print("\n.. NO\n")
                    return pointer, delta, ref_matches, newer_imgs, last_img
                
        elif len(matches.keys()) == 0:
            delta += 1
            print("\n.. NO .. len(matches.keys()) == 0\n")
            return pointer, delta, ref_matches, newer_imgs, last_img

    elif STATIC_IMG_REJECTION_METHOD == 'radiometric':
        im1 = Image.open(IMGS_FROM_SERVER / img1)
        im2 = Image.open(IMGS_FROM_SERVER / img2)
        im1.resize((round(im1.size[0]*1), round(im1.size[1]*1)))
        im2.resize((round(im2.size[0]*1), round(im2.size[1]*1)))
        im1_gray = ImageOps.grayscale(im1)
        im2_gray = ImageOps.grayscale(im2)
        
        mean1 = np.mean(np.array(im1_gray))
        mean2 = np.mean(np.array(im2_gray))

        innovation = np.absolute(mean2 - mean1)
        print("INNOVATION", innovation)

        if innovation > 1:
            if ref_matches == []:
                ref_matches = ["-"]
                shutil.copy(IMGS_FROM_SERVER / "{}".format(img1), KEYFRAMES_DIR / "{}.jpg".format(NextImg(int(last_img))))
                shutil.copy(IMGS_FROM_SERVER / "{}".format(img2), KEYFRAMES_DIR / "{}.jpg".format(NextImg(int(last_img)+1)))
                pointer += 1
                return pointer, delta, ref_matches, newer_imgs, NextImg(int(last_img)+1)
            else:
                shutil.copy(IMGS_FROM_SERVER / "{}".format(img2), KEYFRAMES_DIR / "{}.jpg".format(NextImg(int(last_img))))
                print("\n.. added img\n")
                pointer += 1 + delta
                delta = 0
                newer_imgs = True
                return pointer, delta, ref_matches, newer_imgs, NextImg(int(last_img))
        else:
            delta += 1
            print("\n.. NO\n")
            return pointer, delta, ref_matches, newer_imgs, last_img
    
    else:
        print("Choose 'radiometric' or 'root_sift' as STATIC_IMG_REJECTION_METHOD")
        quit()
