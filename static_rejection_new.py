import logging
from copy import deepcopy

# import os
import shutil

# import subprocess
from pathlib import Path

# from time import time
from typing import List, Tuple, Union

import cv2

# from PIL import Image, ImageOps
import kornia as K
import kornia.feature as KF
import numpy as np
import torch
from easydict import EasyDict as edict
from tqdm import tqdm

from lib.thirdparty.alike.alike import ALike, configs
from lib.thirdparty.transformations import euler_from_matrix
from lib.utils import AverageTimer, timeit

# from icepy.matching.superglue_matcher import SuperGlueMatcher
# from icepy.sfm.two_view_geometry import RelativeOrientation


logger = logging.getLogger("Static Rejection")
logging.basicConfig(
    format="%(asctime)s | %(name)s | %(levelname)s: %(message)s",
    level=logging.INFO,
)

INNOVATION_THRESH = 0.001  # 1.5
INNOVATION_THRESH_PIX = 10
MIN_MATCHES = 50
MIN_POSE_ANGLE = 5  # deg


class AlikeTracker(object):
    """
    A class for tracking similar points in consecutive frames.

    Args:
        viz_res (bool, optional): Whether to visualize the results. Defaults to True.

    Attributes:
        pts_prev (numpy.ndarray): Previous set of points.
        desc_prev (numpy.ndarray): Previous set of descriptors.
        viz_res (bool): Whether to visualize the results.

    Methods:
        track(img, pts, desc): Tracks similar points between consecutive frames.
        mnn_mather(desc1, desc2): Computes the nearest neighbor matches between two sets of descriptors.
        make_plot(img, mpts1, mpts2): Generates a visualization of the matched points.

    """

    def __init__(self, viz_res: bool = True) -> None:
        """
        Initializes the AlikeTracker object.

        Args:
            viz_res (bool, optional): Whether to visualize the results. Defaults to True.
        """
        self.pts_prev = None
        self.desc_prev = None
        self.viz_res = viz_res

    def track(
        self, img: np.ndarray, pts: np.ndarray, desc: np.ndarray
    ) -> Tuple[np.ndarray]:
        """
        Tracks similar points between consecutive frames.

        Args:
            img (numpy.ndarray): Current image.
            pts (numpy.ndarray): Current set of points.
            desc (numpy.ndarray): Current set of descriptors.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, Union[None, numpy.ndarray]]: The matched points from the previous frame,
            the matched points from the current frame, and the visualization of the matched points if self.viz_res is True.

        """
        if self.pts_prev is None:
            self.pts_prev = pts
            self.desc_prev = desc
            mpts1, mpts2 = None, None
            match_fig = deepcopy(img)
        else:
            matches = self.mnn_mather(self.desc_prev, desc)
            mpts1, mpts2 = self.pts_prev[matches[:, 0]], pts[matches[:, 1]]
            if self.viz_res:
                match_fig = self.make_plot(img, mpts1, mpts2)
            else:
                match_fig = None
            self.pts_prev = pts
            self.desc_prev = desc

        return mpts1, mpts2, match_fig

    def mnn_mather(self, desc1: np.ndarray, desc2: np.ndarray) -> np.ndarray:
        """
        Computes the nearest neighbor matches between two sets of descriptors.

        Args:
            desc1 (numpy.ndarray): First set of descriptors.
            desc2 (numpy.ndarray): Second set of descriptors.

        Returns:
            numpy.ndarray: An array of indices indicating the nearest neighbor matches between the two sets of descriptors.

        """
        sim = desc1 @ desc2.transpose()
        sim[sim < 0.9] = 0
        nn12 = np.argmax(sim, axis=1)
        nn21 = np.argmax(sim, axis=0)
        ids1 = np.arange(0, sim.shape[0])
        mask = ids1 == nn21[nn12]
        matches = np.stack([ids1[mask], nn12[mask]])
        return matches.transpose()

    def make_plot(
        self, img: np.ndarray, mpts1: np.ndarray, mpts2: np.ndarray
    ) -> np.ndarray:
        """
        Generates a visualization of the matched points.

        Args:
            img (numpy.ndarray): Current image.
            mpts1 (numpy.ndarray): Matched points from the previous frame.
            mpts2 (numpy.ndarray): Matched points from the current frame.

        Returns:
            numpy.ndarray: An image showing the matched points.

        """
        match_fig = deepcopy(img)
        for pt1, pt2 in zip(mpts1, mpts2):
            p1 = (int(round(pt1[0])), int(round(pt1[1])))
            p2 = (int(round(pt2[0])), int(round(pt2[1])))
            cv2.line(match_fig, p1, p2, (0, 255, 0), lineType=16)
            cv2.circle(match_fig, p2, 1, (0, 0, 255), -1, lineType=16)
        return match_fig


def NextImg(last_img):
    if last_img + 1 < 10:
        next_img = "00000{}".format(last_img + 1)
    elif last_img + 1 < 100:
        next_img = "0000{}".format(last_img + 1)
    elif last_img + 1 < 1000:
        next_img = "000{}".format(last_img + 1)
    elif last_img + 1 < 10000:
        next_img = "00{}".format(last_img + 1)
    elif last_img + 1 < 100000:
        next_img = "0{}".format(last_img + 1)
    elif last_img + 1 < 1000000:
        next_img = "{}".format(last_img + 1)
    return next_img


def process_resize(w, h, resize):
    assert len(resize) > 0 and len(resize) <= 2
    if len(resize) == 1 and resize[0] > -1:
        scale = resize[0] / max(h, w)
        w_new, h_new = int(round(w * scale)), int(round(h * scale))
    elif len(resize) == 1 and resize[0] == -1:
        w_new, h_new = w, h
    else:
        w_new, h_new = resize[0], resize[1]
    return w_new, h_new


def estimate_pose(
    kpts0: np.ndarray,
    kpts1: np.ndarray,
    K0: np.ndarray,
    K1: np.ndarray,
    thresh: float,
    conf=0.9999,
) -> Tuple[np.ndarray]:
    """
    Estimate camera pose given matched points and intrinsics matrix.

    Args:
        kpts0 (np.ndarray): A Nx2 array of keypoints in the first image.
        kpts1 (np.ndarray): A Nx2 array of keypoints in the second image.
        K0 (np.ndarray): A 3x3 intrinsics matrix of the first camera.
        K1 (np.ndarray): A 3x3 intrinsics matrix of the second camera.
        thresh (float): The inlier threshold for RANSAC.
        conf (float, optional): The confidence level for RANSAC. Defaults to 0.9999.

    Returns:
        tuple: A tuple containing the rotation matrix, translation vector, and
        boolean mask indicating inliers.

        - R (np.ndarray): A 3x3 rotation matrix.
        - t (np.ndarray): A 3x1 translation vector.
        - inliers (np.ndarray): A boolean array indicating which keypoints are inliers.
    """
    if len(kpts0) < 5:
        return None

    f_mean = np.mean([K0[0, 0], K1[1, 1], K0[0, 0], K1[1, 1]])
    norm_thresh = thresh / f_mean

    kpts0 = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]

    E, mask = cv2.findEssentialMat(
        kpts0, kpts1, np.eye(3), threshold=norm_thresh, prob=conf, method=cv2.RANSAC
    )

    assert E is not None, "Unable to estimate Essential matrix"

    best_num_inliers = 0
    ret = None
    for _E in np.split(E, len(E) / 3):
        n, R, t, _ = cv2.recoverPose(_E, kpts0, kpts1, np.eye(3), 1e9, mask=mask)
        if n > best_num_inliers:
            best_num_inliers = n
            ret = (R, t[:, 0], mask.ravel() > 0)
    return ret


class StaticRejection:
    def __init__(
        self,
        img_dir: Union[str, Path],
        keyframe_dir: Union[str, Path],
        method: str = "alike",
        matcher_cfg: dict = None,
        camera_matrix: np.ndarray = None,
        resize_to: List[int] = [-1],
        realtime_viz: bool = False,
        viz_res_path: Union[str, Path] = None,
        verbose: bool = False,
    ) -> None:
        
        
        
        self.last_img = 0
        self.img_dir = Path(img_dir)
        assert self.img_dir.is_dir(), f"Invalid image directory {img_dir}"

        self.keyframe_dir = Path(keyframe_dir)
        self.keyframe_dir.mkdir(exist_ok=True, parents=True)
        self.method = method
        self.matcher_cfg = edict(matcher_cfg)
        self.resize_to = resize_to
        self.realtime_viz = realtime_viz
        if viz_res_path is not None:
            self.viz_res_path = Path(viz_res_path)
            self.viz_res_path.mkdir(exist_ok=True)
        else:
            self.viz_res_path = None
        self.verbose = verbose

        self.cur_img_path = None
        self.prev_img_path = None

        self.K = camera_matrix

        # Initialize matching and tracking instances
        if method == "alike":
            # TODO: use a generic configration dictionary as input for StaticRejection class and check dictionary keys for each method.
            self.matcher_cfg = edict(
                {
                    "model": "alike-t",
                    "device": "cuda",
                    "top_k": -1,
                    "scores_th": 0.2,
                    "n_limit": 5000,
                    "subpixel": False,
                }
            )

            self.model = ALike(
                **configs[self.matcher_cfg.model],
                device=self.matcher_cfg.device,
                top_k=self.matcher_cfg.top_k,
                scores_th=self.matcher_cfg.scores_th,
                n_limit=self.matcher_cfg.n_limit,
            )
            self.matcher = AlikeTracker()

    def match_alike(self, cur_img_name: Union[str, Path]) -> Tuple[np.ndarray]:
        self.timer = AverageTimer()
        if self.cur_img_path is None:
            self.cur_img_path = self.img_dir / cur_img_name
            try:
                assert (
                    self.cur_img_path.exists()
                ), f"Current image {cur_img_name} does not exist in image folder"
            except AssertionError as err:
                logging.error(err)
        else:
            self.prev_img_path = self.cur_img_path
            self.cur_img_path = self.img_dir / cur_img_name

        img = cv2.cvtColor(cv2.imread(str(self.cur_img_path)), cv2.COLOR_BGR2RGB)
        if self.resize_to != [-1]:
            assert isinstance(
                resize_to, list
            ), "Invid input for resize_to parameter. It must be a list of integers with the new image dimensions"
            w_new, h_new = process_resize(img.shape[1], img.shape[0], resize=resize_to)
            if any([img.shape[1] > w_new, img.shape[0] > h_new]):
                img = cv2.resize(img, (w_new, h_new))
                if self.verbose:
                    logging.info(f"Images resized to ({w_new},{h_new})")
        self.timer.update("read img")

        pred = self.model(img, sub_pixel=self.matcher_cfg.subpixel)
        self.timer.update("kpts extraction")
        mkpts1, mkpts2, match_img = self.matcher.track(
            img, pred["keypoints"], pred["descriptors"]
        )
        if any([mkpts1 is None, mkpts2 is None]):
            return None
        if len(mkpts1) < MIN_MATCHES:
            if self.verbose:
                logging.error(f"Not enough matches found ({len(mkpts1)}<{MIN_MATCHES})")
            return None

        if self.viz_res_path is not None:
            cv2.imwrite(f"{self.viz_res_path / self.cur_img_path.name}", match_img)
            self.timer.update("export res")

        if self.realtime_viz:
            cv2.setWindowTitle(self.win_name, self.win_name)
            cv2.imshow(self.win_name, match_img)

        if self.verbose:
            self.timer.print(self.method)

        self.compute_innovation(mkpts1, mkpts2)

        return (mkpts1, mkpts2)

    def compute_innovation(self, mkpts1: np.ndarray, mkpts2: np.ndarray) -> None:
        dist = np.linalg.norm(mkpts1 - mkpts2, axis=1)
        median_dist = np.median(dist)
        if median_dist < INNOVATION_THRESH_PIX:
            if self.verbose:
                logging.info(
                    f"Median matching distance {median_dist} < {INNOVATION_THRESH_PIX}: frame rejected."
                )
                return None
        else:
            if self.K is not None:
                threshold = 2
                R, t, valid = estimate_pose(
                    mkpts1,
                    mkpts2,
                    self.K,
                    self.K,
                    thresh=threshold,
                )
                angles = np.array(euler_from_matrix(R))
                max_angle_deg = np.max(np.rad2deg(angles))
                if max_angle_deg < MIN_POSE_ANGLE:
                    if self.verbose:
                        logging.info(
                            f"Larger pose angle {max_angle_deg} < {MIN_POSE_ANGLE}: frame rejected."
                        )
                    return None
            new_name = NextImg(self.last_img) + self.cur_img_path.suffix
            shutil.copy(self.cur_img_path, self.keyframe_dir / new_name)
            self.last_img += 1


if __name__ == "__main__":
    img_dir = "data/MH_01_easy/mav0/cam0/data"
    img_ext = "png"
    keyframe_dir = "colmap_imgs"
    img_dir = Path(img_dir)
    img_list = sorted(img_dir.glob(f"*.{img_ext}"))
    realtime_viz = False
    verbose = False
    resize_to = [-1]

    intrinsics = [458.654, 457.296, 367.215, 248.375]

    Kmat = np.array(
        [
            [intrinsics[0], 0.0, intrinsics[2]],
            [0.0, intrinsics[1], intrinsics[3]],
            [0.0, 0.0, 1.0],
        ]
    )

    static_rej = StaticRejection(
        img_dir,
        keyframe_dir=keyframe_dir,
        method="alike",
        resize_to=resize_to,
        verbose=verbose,
        viz_res_path="matches_plot",
        camera_matrix=Kmat,
        # realtime_viz=False, # Realtime visualization not working
    )

    progress = tqdm(img_list)
    mkpts = {}
    for img in progress:
        mkpts[img.name] = static_rej.match_alike(img.name)

    print("Done")


##### ============= Old code ============ ######

# elif self.method == "superglue":
#     self.matcher_cfg = {
#         "weights": "outdoor",
#         "keypoint_threshold": 0.01,
#         "max_keypoints": 128,
#         "match_threshold": 0.2,
#         "force_cpu": False,
#     }
#     self.matcher = SuperGlueMatcher(self.matcher_cfg)

# elif method == "loftr":
#     self.matcher_cfg = edict(
#         {
#             "device": "cuda",
#         }
#     )
#     device = torch.device(self.matcher_cfg.device)
#     self.matcher = KF.LoFTR(pretrained="outdoor").to(device).eval()

# else:
#     raise ValueError("Inalid input method")


# def load_torch_image(
#     fname: Union[str, Path],
#     device=torch.device("cpu"),
#     resize_to: Tuple[int] = [-1],
#     as_grayscale: bool = True,
#     as_float: bool = True,
# ):
#     fname = str(fname)
#     timg = K.image_to_tensor(cv2.imread(fname), False)
#     timg = K.color.bgr_to_rgb(timg.to(device))

#     if as_float:
#         timg = timg.float() / 255.0

#     if as_grayscale:
#         timg = K.color.rgb_to_grayscale(timg)

#     h0, w0 = timg.shape[2:]

#     if resize_to != [-1] and resize_to[0] > w0:
#         timg = K.geometry.resize(timg, size=resize_to, antialias=True)
#         h1, w1 = timg.shape[2:]
#         resize_ratio = (float(w0) / float(w1), float(h0) / float(h1))
#     else:
#         resize_ratio = (1.0, 1.0)

#     return timg, resize_ratio


# def static_rejection(
#     img_dir: Union[str, Path],
#     cur_img_name: Union[str, Path],
#     prev_img_name: Union[str, Path],
#     keyframe_dir: Union[str, Path],
#     method: str = "superglue",
#     resize_to: List[int] = [-1],
#     verbose: bool = False,
# ):
#     timer = AverageTimer()

#     img_dir = Path(img_dir)
#     cur_img_name = Path(cur_img_name)
#     prev_img_name = Path(prev_img_name)
#     cur_img = img_dir / cur_img_name
#     prev_img = img_dir / prev_img_name
#     assert img_dir.is_dir(), f"Invalid image directory {img_dir}"
#     # Check if the two images exists, otherwise skip.
#     try:
#         assert (
#             cur_img.exists()
#         ), f"Current image {cur_img_name} does not exist in image folder"
#         assert (
#             prev_img.exists()
#         ), f"Previous image {prev_img_name} does not exist in image folder"
#     except AssertionError as err:
#         print("!! Frame truncated !!")
#         return None

#     if method == "superglue":
#         im1 = cv2.imread(str(cur_img), flags=cv2.IMREAD_GRAYSCALE)
#         im2 = cv2.imread(str(prev_img), flags=cv2.IMREAD_GRAYSCALE)
#         timer.update("loaded imgs")

#         if resize_to != [-1]:
#             assert isinstance(
#                 resize_to, list
#             ), "Invid input for resize_to parameter. It must be a list of integers with the new image dimensions"
#             w_new, h_new = process_resize(im1.shape[1], im1.shape[0], resize=resize_to)
#             if any([im1.shape[1] > w_new, im1.shape[0] > h_new]):
#                 im1 = cv2.resize(im1, (w_new, h_new))
#                 im2 = cv2.resize(im2, (w_new, h_new))
#                 if verbose:
#                     logging.info(f"Images resized to ({w_new},{h_new})")
#             timer.update("resizing")

#         suerglue_cfg = {
#             "weights": "outdoor",
#             "keypoint_threshold": 0.01,
#             "max_keypoints": 128,
#             "match_threshold": 0.2,
#             "force_cpu": False,
#         }
#         matcher = SuperGlueMatcher(suerglue_cfg)
#         mkpts = matcher.match(np.asarray(im1), np.asarray(im2))
#         timer.update("matching")
#         timer.print(f"Static rejection {method}")

#         # mkpts = matcher.geometric_verification(
#         #     threshold=2,
#         #     confidence=0.99,
#         #     symmetric_error_check=False,
#         # )
#         # matcher.viz_matches("test.png")

#     elif method == "loftr":
#         device = torch.device("cuda")
#         timg0, res_ratio0 = load_torch_image(
#             cur_img, device=device, resize_to=resize_to, as_grayscale=True
#         )
#         timg1, res_ratio1 = load_torch_image(
#             prev_img, device=device, resize_to=resize_to, as_grayscale=True
#         )
#         matcher = KF.LoFTR(pretrained="outdoor").to(device).eval()

#         with torch.inference_mode():
#             input_dict = {"image0": timg0, "image1": timg1}
#             correspondences = matcher(input_dict)
#     if method == "alike":
#         args = edict(
#             {
#                 "model": "alike-t",
#                 "device": "cuda",
#                 "top_k": -1,
#                 "scores_th": 0.2,
#                 "n_limit": 5000,
#                 "subpixel": False,
#             }
#         )

#         model = ALike(
#             **configs[args.model],
#             device=args.device,
#             top_k=args.top_k,
#             scores_th=args.scores_th,
#             n_limit=args.n_limit,
#         )
#         tracker = AlikeTracker()
#         im1 = cv2.cvtColor(cv2.imread(str(cur_img)), cv2.COLOR_BGR2RGB)
#         im2 = cv2.cvtColor(cv2.imread(str(prev_img)), cv2.COLOR_BGR2RGB)

#         pred = model(im1, sub_pixel=args.subpixel)
#         _, _ = tracker.update(im1, pred["keypoints"], pred["descriptors"])

#         pred = model(im2, sub_pixel=args.subpixel)
#         out, N_matches = tracker.update(im2, pred["keypoints"], pred["descriptors"])

#         return tracker.matches
#         # print("Done.")
