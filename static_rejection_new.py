import logging
from copy import deepcopy

# import os
# import shutil
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
from lib.utils import AverageTimer, timeit

# from icepy.matching.superglue_matcher import SuperGlueMatcher
# from icepy.sfm.two_view_geometry import RelativeOrientation


logger = logging.getLogger("Static Rejection")
logging.basicConfig(
    format="%(asctime)s | %(name)s | %(levelname)s: %(message)s",
    level=logging.INFO,
)

INNOVATION_THRESH = 0.001  # 1.5


class AlikeTracker(object):
    def __init__(self):
        self.pts_prev = None
        self.desc_prev = None

    def update(self, img, pts, desc):
        self.matches = None
        n_matches = 0
        if self.pts_prev is None:
            self.pts_prev = pts
            self.desc_prev = desc
            match_fig = deepcopy(img)
            for pt1 in pts:
                p1 = (int(round(pt1[0])), int(round(pt1[1])))
                cv2.circle(match_fig, p1, 1, (0, 0, 255), -1, lineType=16)
        else:
            self.matches = self.mnn_mather(self.desc_prev, desc)
            mpts1, mpts2 = self.pts_prev[self.matches[:, 0]], pts[self.matches[:, 1]]
            n_matches = len(self.matches)

            match_fig = deepcopy(img)
            for pt1, pt2 in zip(mpts1, mpts2):
                p1 = (int(round(pt1[0])), int(round(pt1[1])))
                p2 = (int(round(pt2[0])), int(round(pt2[1])))
                cv2.line(match_fig, p1, p2, (0, 255, 0), lineType=16)
                cv2.circle(match_fig, p2, 1, (0, 0, 255), -1, lineType=16)

            self.pts_prev = pts
            self.desc_prev = desc

        return self.matches, match_fig, n_matches

    def mnn_mather(self, desc1, desc2):
        sim = desc1 @ desc2.transpose()
        sim[sim < 0.9] = 0
        nn12 = np.argmax(sim, axis=1)
        nn21 = np.argmax(sim, axis=0)
        ids1 = np.arange(0, sim.shape[0])
        mask = ids1 == nn21[nn12]
        matches = np.stack([ids1[mask], nn12[mask]])
        return matches.transpose()


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


def load_torch_image(
    fname: Union[str, Path],
    device=torch.device("cpu"),
    resize_to: Tuple[int] = [-1],
    as_grayscale: bool = True,
    as_float: bool = True,
):
    fname = str(fname)
    timg = K.image_to_tensor(cv2.imread(fname), False)
    timg = K.color.bgr_to_rgb(timg.to(device))

    if as_float:
        timg = timg.float() / 255.0

    if as_grayscale:
        timg = K.color.rgb_to_grayscale(timg)

    h0, w0 = timg.shape[2:]

    if resize_to != [-1] and resize_to[0] > w0:
        timg = K.geometry.resize(timg, size=resize_to, antialias=True)
        h1, w1 = timg.shape[2:]
        resize_ratio = (float(w0) / float(w1), float(h0) / float(h1))
    else:
        resize_ratio = (1.0, 1.0)

    return timg, resize_ratio


class StaticRejection:
    def __init__(
        self,
        img_dir: Union[str, Path],
        keyframe_dir: Union[str, Path],
        method: str = "alike",
        matcher_cfg: dict = None,
        resize_to: List[int] = [-1],
        viz_res_path: Union[str, Path] = None,
        verbose: bool = False,
    ) -> None:
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

        # Initialize matching and tracking instances
        if method == "alike":
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

        # elif self.method == "superglue":
        #     # TODO: use a generic configration dictionary as input for StaticRejection class and check dictionary keys for each method.

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

    def match_alike(self, cur_img_name: Union[str, Path]):
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
        matches, match_img, n_matches = self.matcher.update(
            img, pred["keypoints"], pred["descriptors"]
        )
        if self.viz_res_path is not None:
            cv2.imwrite(f"{self.viz_res_path / self.cur_img_path.name}", match_img)
            self.timer.update("export res")

        if self.verbose:
            self.timer.print(self.method)

        return matches, match_img


if __name__ == "__main__":
    img_dir = "data/MH_01_easy/mav0/cam0/data"
    img_ext = "png"
    keyframe_dir = "colmap_imgs"
    img_dir = Path(img_dir)
    img_list = sorted(img_dir.glob(f"*.{img_ext}"))
    realtime_viz = False
    verbose = False
    resize_to = [-1]

    if realtime_viz:
        win_name = "alike matching"
        cv2.namedWindow(win_name)

    static_rej = StaticRejection(
        img_dir,
        keyframe_dir=keyframe_dir,
        method="alike",
        resize_to=resize_to,
        verbose=verbose,
        viz_res_path="matches_plot",
    )

    progress = tqdm(img_list)
    mkpts = {}
    for img in progress:
        mkpts[img.name], match_img = static_rej.match_alike(img.name)
        if realtime_viz:
            cv2.setWindowTitle(win_name, "alike")
            cv2.imshow(win_name, match_img)
            if cv2.waitKey(1) == ord("q"):
                break

    print("Done")


##### ============= Old code ============ ######

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
