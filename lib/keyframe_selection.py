import os
import cv2
import sys
import numpy as np
import shutil
import random

from lib import (LocalFeatures, Matcher, keyframe, utils)

INNOVATION_THRESH_PIX = 120 #200
RANSAC_THRESHOLD = 10

def KeyframeSelection(
    KFS_METHOD,
    KFS_LOCAL_FEATURE,
    KFS_N_FEATURES,
    img1,
    img2,
    IMGS_FROM_SERVER,
    KEYFRAMES_DIR,
    keyframes_list,
    pointer, 
    delta
    ):

    if KFS_METHOD == 'local_features':
        local_feature = LocalFeatures.LocalFeatures([img1, img2], IMGS_FROM_SERVER, KFS_N_FEATURES, KFS_LOCAL_FEATURE)
        if KFS_LOCAL_FEATURE == 'ORB':
            all_keypoints, all_descriptors = local_feature.ORB()
        elif KFS_LOCAL_FEATURE == 'ALIKE':
            print("TO BE IMPLEMENTED")

        desc1 = all_descriptors[0]
        desc2 = all_descriptors[1]
        kpts1 = all_keypoints[0]
        kpts2 = all_keypoints[1]

        matcher = Matcher.Matcher(desc1, desc2)
        # Here we should handle that we can use different kinds of matcher (also adding the option in config file)
        matches = matcher.mnn_matcher_cosine()
        matches_im1 = matches[:,0]
        matches_im2 = matches[:,1]

        mpts1 = kpts1[matches_im1]
        mpts2 = kpts2[matches_im2]

        match_dist = np.linalg.norm(mpts1 - mpts2, axis=1)
        median_match_dist = np.median(match_dist)

        ### Ransac to eliminate outliers
        #TODO: move RANSAC to a separate function (and possible allow choises to use other method than ransac, eg. pydegensac, with same interface)
        rands = []
        scores = []
        for i in range(100):
            rand = random.randrange(0, len(mpts1[0]))
            reference_distance = np.linalg.norm(mpts1[rand] - mpts2[rand])
            score = np.sum(np.absolute(match_dist - reference_distance) < RANSAC_THRESHOLD) / len(match_dist)
            rands.append(rand)
            scores.append(score)

        max_consensus = rands[np.argmax(scores)]
        reference_distance = np.linalg.norm(mpts1[max_consensus] - mpts2[max_consensus])
        mask = np.absolute(match_dist - reference_distance) > RANSAC_THRESHOLD
      

        match_dist = np.linalg.norm(mpts1 - mpts2, axis=1)
        median_match_dist = np.median(match_dist)
        print("median_match_dist", median_match_dist)

        if median_match_dist > INNOVATION_THRESH_PIX:
            existing_keyframe_number = len(os.listdir(KEYFRAMES_DIR))
            shutil.copy(
                IMGS_FROM_SERVER / "{}".format(img2),
                KEYFRAMES_DIR / "{}".format(utils.Id2name(existing_keyframe_number)))
            camera_id=1
            new_keyframe = keyframe.Keyframe(img2, existing_keyframe_number, utils.Id2name(existing_keyframe_number), camera_id, pointer+delta+1)
            keyframes_list.append(new_keyframe)
            print("new_keyframe.image_id", new_keyframe.image_id)

            pointer += 1 + delta
            delta = 0

        else:
            print("Frame rejected")
            delta += 1
    
    else:
        # Here we can implement methods like LoFTR
        print("Error! Only local_features method is implemented")
        quit()

    return keyframes_list, pointer, delta