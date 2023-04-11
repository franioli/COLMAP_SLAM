import logging
import os
import sys
import cv2
import random
import shutil
from pathlib import Path
from typing import List, Union
from copy import deepcopy

import numpy as np
from easydict import EasyDict as edict

from lib import utils
from lib.keyframes import KeyFrame, KeyFrameList
from lib.local_features import LocalFeatures
from lib.matching import Matcher
from lib.thirdparty.alike.alike import ALike, configs


INNOVATION_THRESH_PIX = 120  # 200
MIN_MATCHES = 20
RANSAC_THRESHOLD = 10
RANSAC_ITERATIONS = 1000

# TODO: use logger instead of print
logger = logging.getLogger(__name__)


# TODO: make ransac function independent from KeyFrameSelector class
def ransac(
    kpts1: np.ndarray, kpts2: np.ndarray, threshold: float = RANSAC_THRESHOLD
) -> np.ndarray:
    pass


# TODO: integrate ALike in LocalFeatures (see self.extract_features() method)
class KeyFrameSelector:
    def __init__(
        self,
        keyframes_list: List,  # KeyFrameList,
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
        verbose: bool = False,
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
        # TODO: validate input parameters
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
        if self.realtime_viz:
            cv2.namedWindow(self.method)

        if viz_res_path is not None:
            self.viz_res_path = Path(viz_res_path)
            self.viz_res_path.mkdir(exist_ok=True)
        else:
            self.viz_res_path = None
        self.timer = utils.AverageTimer()
        self.verbose = verbose

        # Set initial images to None
        self.img1 = None
        self.img2 = None
        self.match_img = None

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
        self.timer.update("init class")

        # If Local feeature is a Deep Learning model, we need to load it
        if self.local_feature == "ALIKE":
            # TODO: validate that the config dictionary self.matcher_cfg is correct and containes all the required keys
            self.model = ALike(
                **configs[self.local_feature_cfg.model],
                device=self.local_feature_cfg.device,
                top_k=self.local_feature_cfg.top_k,
                scores_th=self.local_feature_cfg.scores_th,
                n_limit=self.local_feature_cfg.n_limit,
            )
            self.timer.update("load model")

    def clear_matches(self) -> None:
        # Now it is executed for safety reasons vy self.run() at the end of the keyframeselection process... it may be removed if we want to keep track of the previous matched
        self.kpts1 = None
        self.kpts2 = None
        self.desc1 = None
        self.desc2 = None
        self.mpts1 = None
        self.mpts2 = None

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
            # Now Alike is implemented independentely from LocalFeatures because the Alike model is loaded only once when the KeyFrameSelector is initialized for speedup the procedure.
            # TODO: implement Alike as a LocalFeatures method, so that we can use it in the same way as ORB
            # TODO: to speed up the procedure, keep the keypoints and descriptors of the last keyframe in memory, instead of extracting them again
            img1 = cv2.cvtColor(cv2.imread(str(self.img1)), cv2.COLOR_BGR2RGB)
            features1 = self.model(img1, sub_pixel=self.local_feature_cfg.subpixel)
            self.kpts1 = features1["keypoints"]
            self.desc1 = features1["descriptors"]

            img2 = cv2.cvtColor(cv2.imread(str(self.img2)), cv2.COLOR_BGR2RGB)
            features2 = self.model(img2, sub_pixel=self.local_feature_cfg.subpixel)
            self.kpts2 = features2["keypoints"]
            self.desc2 = features2["descriptors"]
        else:
            # Here we can implement methods like LoFTR
            logger.error("Error! Only local_features method is implemented")
            quit()

        self.timer.update("features extraction")

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

            if self.realtime_viz is True or self.viz_res_path is not None:
                img = cv2.imread(str(self.img2))
                self.match_img = matcher.make_plot(img, self.mpts1, self.mpts2)

        else:
            # Here we can implement matching methods
            logger.error("Error! Only mnn_cosine method is implemented")
            quit()

        ### Ransac to eliminate outliers
        # TODO: move RANSAC to a separate function (and possible allow choises to use other method than ransac, eg. pydegensac, with same interface)
        if self.geometric_verification == "ransac":
            match_dist = np.linalg.norm(self.mpts1 - self.mpts2, axis=1)
            rands = []
            scores = []
            for i in range(RANSAC_ITERATIONS):
                rand = random.randrange(0, self.mpts1.shape[0])
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
            logger.info(
                f"Ransac found {len(list(filter(None, mask)))}/{len(mask)} inliers"
            )
            self.mpts1 = self.mpts1[mask, :]
            self.mpts2 = self.mpts2[mask, :]
        else:
            # Here we can implement other methods
            logger.error("Error! Only ransac method is implemented")
            quit()

        self.timer.update("matching")
        return True

    def innovation_check(self) -> bool:
        match_dist = np.linalg.norm(self.mpts1 - self.mpts2, axis=1)
        median_match_dist = np.median(match_dist)
        logger.info(f"median_match_dist: {median_match_dist:.2f}")

        if len(self.mpts1) < MIN_MATCHES:
            logger.info("Frame rejected: not enogh matches")
            self.delta += 1
            self.timer.update("innovation check")

            return False

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
            # self.keyframes_list.append(new_keyframe)
            logger.info(
                f"Frame accepted. New_keyframe image_id: {new_keyframe.image_id}"
            )

            self.pointer += 1 + self.delta
            self.delta = 0
            self.timer.update("innovation check")

            return True

        else:
            logger.info("Frame rejected")
            self.delta += 1
            self.timer.update("innovation check")

            return False

    def run(self, img1: Union[str, Path], img2: Union[str, Path]):
        if not self.extract_features(img1, img2):
            raise RuntimeError("Error in extract_features")
        if not self.match_features():
            raise RuntimeError("Error in match_features")
        keyframe_accepted = self.innovation_check()

        if self.match_img is not None:
            if self.viz_res_path is not None:
                cv2.imwrite(str(self.viz_res_path / self.img2.name), self.match_img)
            if self.realtime_viz:
                if keyframe_accepted:
                    win_name = self.method + ": Keyframe accepted"
                else:
                    win_name = self.method + ": Frame rejected"
                cv2.setWindowTitle(self.method, win_name)
                cv2.imshow(self.method, self.match_img)
                if cv2.waitKey(1) == ord("q"):
                    sys.exit()

        self.clear_matches()
        if self.verbose:
            self.timer.print()

        return self.keyframes_list, self.pointer, self.delta


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
            new_keyframe = KeyFrame(
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
    img0 = Path("data/MH_01_easy/mav0/cam0/data/1403636579763555584.png")
    img1 = Path("data/MH_01_easy/mav0/cam0/data/1403636587363555584.png")

    keyframes_list = KeyFrameList()
    processed_imgs = []
    oriented_imgs_batch = []
    pointer = 0  # pointer points to the last oriented image
    delta = 0  # delta is equal to the number of processed but not oriented imgs

    timer = utils.AverageTimer()

    # Test ORB
    kfs = KeyFrameSelector(
        keyframes_list=keyframes_list,
        last_keyframe_pointer=pointer,
        last_keyframe_delta=delta,
        local_feature="ORB",
        realtime_viz=False,
    )
    (
        keyframes_list,
        pointer,
        delta,
    ) = kfs.run(img0, img1)
    timer.update("orb")

    # Test ALIKE
    alike_cfg = edict(
        {
            "model": "alike-t",
            "device": "cuda",
            "top_k": 512,
            "scores_th": 0.2,
            "n_limit": 5000,
            "subpixel": False,
        }
    )
    kfs = KeyFrameSelector(
        keyframes_list=keyframes_list,
        last_keyframe_pointer=pointer,
        last_keyframe_delta=delta,
        local_feature="ALIKE",
        local_feature_cfg=alike_cfg,
        realtime_viz=False,
    )
    (
        keyframes_list,
        pointer,
        delta,
    ) = kfs.run(img0, img1)
    timer.update("alike")

    timer.print()

    # cv2.destroyAllWindows()

    print("Done")
