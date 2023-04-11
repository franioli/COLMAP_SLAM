import logging
import os
import random
import shutil
from pathlib import Path
from typing import List, Union

import numpy as np
from easydict import EasyDict as edict

from lib import utils
from lib.keyframes import KeyFrame, KeyFrameList
from lib.local_features import LocalFeatures
from lib.matching import Matcher

LOG_LEVEL = logging.INFO
logging.basicConfig(
    format="%(asctime)s | %(name)s | %(levelname)s: %(message)s",
    level=LOG_LEVEL,
)

INNOVATION_THRESH_PIX = 120  # 200
RANSAC_THRESHOLD = 10

# TODO: use logging instead of print


def ransac(
    kpts1: np.ndarray, kpts2: np.ndarray, threshold: float = RANSAC_THRESHOLD
) -> np.ndarray:
    pass


class KeyFrameSelector:
    def __init__(
        self,
        keyframes_list: KeyFrameList,
        last_keyframe_pointer: int,
        last_keyframe_delta: int,
        keyframes_dir: Union[str, Path] = "colmap_imgs",
        kfs_method: Union[str, Path] = "local_features",
        local_feature: str = "ORB",
        local_feature_cfg: dict = None,
        n_features: int = 512,
        kfs_matcher: str = "mnn_cosine",
        geometric_verification: str = "ransac",
        realtime_viz: bool = False,
        viz_res_path: Union[str, Path] = None,
    ) -> None:
        """
        __init__ _summary_

        Args:
            keyframes_list (List[Keyframe]): _description_
            last_keyframe_pointer (int): _description_
            last_keyframe_delta (int): _description_
            keyframes_dir (Union[str, Path], optional): _description_. Defaults to "colmap_imgs".
            kfs_method (Union[str, Path], optional): _description_. Defaults to "local_features".
            local_feature (str, optional): _description_. Defaults to "ORB".
            local_feature_cfg (dict, optional): _description_. Defaults to None.
            n_features (int, optional): _description_. Defaults to 512.
            kfs_matcher (str, optional): _description_. Defaults to "mnn_cosine".
            geometric_verification (str, optional): _description_. Defaults to "ransac".
            realtime_viz (bool, optional): _description_. Defaults to False.
            viz_res_path (Union[str, Path], optional): _description_. Defaults to None.
        """
        self.keyframes_list = keyframes_list
        self.keyframes_dir = Path(keyframes_dir)
        self.keyframes_dir.mkdir(exist_ok=True, parents=True)
        self.method = kfs_method
        self.local_feature = local_feature
        self.local_feature_cfg = edict(local_feature_cfg)
        self.n_features = n_features
        self.matcher = kfs_matcher
        self.geometric_verification = geometric_verification
        self.realtime_viz = realtime_viz
        self.viz_res_path = viz_res_path

        # Set initial images to None
        self.img1 = None
        self.img2 = None

        # Set initial keyframes, descr, mpts to None
        self.kpts1 = None
        self.kpts2 = None
        self.desc1 = None
        self.desc2 = None
        self.mpts1 = None
        self.mpts2 = None

        # Set initial pointer and delta
        self.pointer = last_keyframe_pointer
        self.delta = last_keyframe_delta

    def extract_features(self, img1: Union[str, Path], img2: Union[str, Path]) -> bool:
        self.img1 = Path(img1)
        self.img2 = Path(img2)

        if self.method == "local_features":
            local_feature = LocalFeatures(
                [img1, img2], self.n_features, self.local_feature
            )
        if self.local_feature == "ORB":
            all_keypoints, all_descriptors = local_feature.ORB()
            self.kpts1 = all_keypoints[0]
            self.kpts2 = all_keypoints[1]
            self.desc1 = all_descriptors[0]
            self.desc2 = all_descriptors[1]

        elif self.local_feature == "ALIKE":
            logging.error("TO BE IMPLEMENTED")
        else:
            # Here we can implement methods like LoFTR
            logging.error("Error! Only local_features method is implemented")
            quit()
        return True

    def match_features(self) -> bool:
        assert all(
            [self.kpts1 is not None, self.kpts2 is not None]
        ), "kpts1 or kpts2 is None, run extract_features first"

        # Here we should handle that we can use different kinds of matcher (also adding the option in config file)
        if self.matcher == "mnn_cosine":
            matcher = Matcher(self.desc1, self.desc2)
            matches = matcher.mnn_matcher_cosine()
            matches_im1 = matches[:, 0]
            matches_im2 = matches[:, 1]
            self.mpts1 = self.kpts1[matches_im1]
            self.mpts2 = self.kpts2[matches_im2]

        else:
            # Here we can implement matching methods
            logging.error("Error! Only mnn_cosine method is implemented")
            quit()

        ### Ransac to eliminate outliers
        # TODO: move RANSAC to a separate function (and possible allow choises to use other method than ransac, eg. pydegensac, with same interface)
        if self.geometric_verification == "ransac":
            match_dist = np.linalg.norm(self.mpts1 - self.mpts2, axis=1)
            rands = []
            scores = []
            for i in range(100):
                rand = random.randrange(0, len(self.mpts1[0]))
                reference_distance = np.linalg.norm(self.mpts1[rand] - self.mpts2[rand])
                score = np.sum(
                    np.absolute(match_dist - reference_distance) < RANSAC_THRESHOLD
                ) / len(match_dist)
                rands.append(rand)
                scores.append(score)
                max_consensus = rands[np.argmax(scores)]
                reference_distance = np.linalg.norm(
                    self.mpts1[max_consensus] - self.mpts2[max_consensus]
                )
                mask = np.absolute(match_dist - reference_distance) > RANSAC_THRESHOLD

                match_dist = np.linalg.norm(self.mpts1 - self.mpts2, axis=1)
            # logging.info("ransac rsults: ...")
        else:
            # Here we can implement other methods
            logging.error("Error! Only ransac method is implemented")
            quit()
        return True

    def innovation_check(self) -> bool:
        match_dist = np.linalg.norm(self.mpts1 - self.mpts2, axis=1)
        median_match_dist = np.median(match_dist)
        logging.info(f"median_match_dist: {median_match_dist:.2f}")

        if median_match_dist > INNOVATION_THRESH_PIX:
            existing_keyframe_number = len(os.listdir(self.keyframes_dir))
            shutil.copy(
                self.img2,
                self.keyframes_dir / f"{utils.Id2name(existing_keyframe_number)}",
            )
            camera_id = 1
            new_keyframe = KeyFrame(
                self.img2,
                existing_keyframe_number,
                utils.Id2name(existing_keyframe_number),
                camera_id,
                self.pointer + self.delta + 1,
            )
            self.keyframes_list.add_keyframe(new_keyframe)
            logging.info(f"new_keyframe.image_id: {new_keyframe.image_id}")

            self.pointer += 1 + self.delta
            self.delta = 0

            return True

        else:
            logging.info("Frame rejected")
            self.delta += 1

            return False

    def run(self, img1: Union[str, Path], img2: Union[str, Path]):
        if not self.extract_features(img1, img2):
            raise RuntimeError("Error in extract_features")
        if not self.match_features():
            raise RuntimeError("Error in match_features")
        keyframe_accepted = self.innovation_check()

        return self.keyframes_list, self.pointer, self.delta


# def KeyframeSelection(
#     KFS_METHOD,
#     KFS_LOCAL_FEATURE,
#     KFS_N_FEATURES,
#     img1,
#     img2,
#     KEYFRAMES_DIR,
#     keyframes_list,
#     pointer,
#     delta,
# ):
#     if KFS_METHOD == "local_features":
#         local_feature = LocalFeatures([img1, img2], KFS_N_FEATURES, KFS_LOCAL_FEATURE)
#         if KFS_LOCAL_FEATURE == "ORB":
#             all_keypoints, all_descriptors = local_feature.ORB()
#         elif KFS_LOCAL_FEATURE == "ALIKE":
#             print("TO BE IMPLEMENTED")

#         desc1 = all_descriptors[0]
#         desc2 = all_descriptors[1]
#         kpts1 = all_keypoints[0]
#         kpts2 = all_keypoints[1]

#         matcher = Matcher(desc1, desc2)
#         # Here we should handle that we can use different kinds of matcher (also adding the option in config file)
#         matches = matcher.mnn_matcher_cosine()
#         matches_im1 = matches[:, 0]
#         matches_im2 = matches[:, 1]

#         mpts1 = kpts1[matches_im1]
#         mpts2 = kpts2[matches_im2]

#         match_dist = np.linalg.norm(mpts1 - mpts2, axis=1)
#         median_match_dist = np.median(match_dist)

#         ### Ransac to eliminate outliers
#         # TODO: move RANSAC to a separate function (and possible allow choises to use other method than ransac, eg. pydegensac, with same interface)
#         rands = []
#         scores = []
#         for i in range(100):
#             rand = random.randrange(0, len(mpts1[0]))
#             reference_distance = np.linalg.norm(mpts1[rand] - mpts2[rand])
#             score = np.sum(
#                 np.absolute(match_dist - reference_distance) < RANSAC_THRESHOLD
#             ) / len(match_dist)
#             rands.append(rand)
#             scores.append(score)

#         max_consensus = rands[np.argmax(scores)]
#         reference_distance = np.linalg.norm(mpts1[max_consensus] - mpts2[max_consensus])
#         mask = np.absolute(match_dist - reference_distance) > RANSAC_THRESHOLD

#         match_dist = np.linalg.norm(mpts1 - mpts2, axis=1)
#         median_match_dist = np.median(match_dist)
#         print("median_match_dist", median_match_dist)

#         if median_match_dist > INNOVATION_THRESH_PIX:
#             existing_keyframe_number = len(os.listdir(KEYFRAMES_DIR))
#             shutil.copy(
#                 img2,
#                 KEYFRAMES_DIR / f"{utils.Id2name(existing_keyframe_number)}",
#             )
#             camera_id = 1
#             new_keyframe = Keyframe(
#                 img2,
#                 existing_keyframe_number,
#                 utils.Id2name(existing_keyframe_number),
#                 camera_id,
#                 pointer + delta + 1,
#             )
#             keyframes_list.append(new_keyframe)
#             print("new_keyframe.image_id", new_keyframe.image_id)

#             pointer += 1 + delta
#             delta = 0

#         else:
#             print("Frame rejected")
#             delta += 1

#     else:
#         # Here we can implement methods like LoFTR
#         print("Error! Only local_features method is implemented")
#         quit()

#     return keyframes_list, pointer, delta


if __name__ == "__main__":
    img0 = Path("imgs/1403636579763555584.jpg")
    img1 = Path("imgs/1403636591263555584.jpg")

    keyframes_list = KeyFrameList()
    processed_imgs = []
    oriented_imgs_batch = []
    pointer = 0  # pointer points to the last oriented image
    delta = 0  # delta is equal to the number of processed but not oriented imgs

    kfs = KeyFrameSelector(
        keyframes_list=keyframes_list,
        last_keyframe_pointer=pointer,
        last_keyframe_delta=delta,
        keyframes_dir=Path("keyframes"),
    )
    (
        keyframes_list,
        pointer,
        delta,
    ) = kfs.run(img0, img1)

    print("Done")
