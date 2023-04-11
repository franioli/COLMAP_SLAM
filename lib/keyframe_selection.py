import os
import cv2
import sys
import numpy as np
import shutil
import random
from typing import Union
from pathlib import Path

from lib.keyframe import Keyframe
from lib.matching import Matcher
from lib.local_features import LocalFeatures
from lib import utils


INNOVATION_THRESH_PIX = 120  # 200
RANSAC_THRESHOLD = 10


class KeyFrameSelector:
    def __init__(
        self,
        kfs_method: Union[str, Path] = "local_features",
        kfd_local_feature: str = "ORB",
        kfs_n_features: int = 512,
        keyframe_dir: Union[str, Path] = "colmap_imgs",
    ) -> None:
        self.kfs_method = kfs_method
        self.kfd_local_feature = kfd_local_feature
        self.kfs_n_features = kfs_n_features
        self.keyframe_dir = keyframe_dir

        # Set initial pointer and delta to 0
        self.pointer = 0
        self.delta = 0

    def run(self, img1: Union[str, Path], img2: Union[str, Path]):
        self.img1 = Path(img1)
        self.img2 = Path(img2)

        if self.kfs_method == "local_features":
            local_feature = LocalFeatures(
                [img1, img2], self.kfs_n_features, self.kfd_local_feature
            )
        if self.kfd_local_feature == "ORB":
            all_keypoints, all_descriptors = local_feature.ORB()
        elif self.kfd_local_feature == "ALIKE":
            print("TO BE IMPLEMENTED")
        else:
            # Here we can implement methods like LoFTR
            print("Error! Only local_features method is implemented")
            quit()


def KeyframeSelection(
    KFS_METHOD,
    KFS_LOCAL_FEATURE,
    KFS_N_FEATURES,
    img1,
    img2,
    KEYFRAMES_DIR,
    keyframes_list,
    pointer,
    delta,
):
    if KFS_METHOD == "local_features":
        local_feature = LocalFeatures([img1, img2], KFS_N_FEATURES, KFS_LOCAL_FEATURE)
        if KFS_LOCAL_FEATURE == "ORB":
            all_keypoints, all_descriptors = local_feature.ORB()
        elif KFS_LOCAL_FEATURE == "ALIKE":
            print("TO BE IMPLEMENTED")

        desc1 = all_descriptors[0]
        desc2 = all_descriptors[1]
        kpts1 = all_keypoints[0]
        kpts2 = all_keypoints[1]

        matcher = Matcher(desc1, desc2)
        # Here we should handle that we can use different kinds of matcher (also adding the option in config file)
        matches = matcher.mnn_matcher_cosine()
        matches_im1 = matches[:, 0]
        matches_im2 = matches[:, 1]

        mpts1 = kpts1[matches_im1]
        mpts2 = kpts2[matches_im2]

        match_dist = np.linalg.norm(mpts1 - mpts2, axis=1)
        median_match_dist = np.median(match_dist)

        ### Ransac to eliminate outliers
        # TODO: move RANSAC to a separate function (and possible allow choises to use other method than ransac, eg. pydegensac, with same interface)
        rands = []
        scores = []
        for i in range(100):
            rand = random.randrange(0, len(mpts1[0]))
            reference_distance = np.linalg.norm(mpts1[rand] - mpts2[rand])
            score = np.sum(
                np.absolute(match_dist - reference_distance) < RANSAC_THRESHOLD
            ) / len(match_dist)
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
                img2,
                KEYFRAMES_DIR / f"{utils.Id2name(existing_keyframe_number)}",
            )
            camera_id = 1
            new_keyframe = Keyframe(
                img2,
                existing_keyframe_number,
                utils.Id2name(existing_keyframe_number),
                camera_id,
                pointer + delta + 1,
            )
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


if __name__ == "__main__":
    img0 = Path("imgs/1403636579763555584.jpg")
    img1 = Path("imgs/1403636580263555584.jpg")

    kfs = KeyFrameSelector()
    kfs.run(img0, img1)
