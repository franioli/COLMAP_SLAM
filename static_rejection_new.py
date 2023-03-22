import cv2
import logging
import numpy as np
import os
import shutil
import subprocess
from pathlib import Path
from time import time
from typing import List, Tuple, Union
from PIL import Image, ImageOps
import kornia as K
import kornia.feature as KF
import torch

from icepy.matching.superglue_matcher import SuperGlueMatcher

# from icepy.sfm.two_view_geometry import RelativeOrientation

from lib.utils import timeit, AverageTimer


INNOVATION_THRESH = 0.001  # 1.5


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


@timeit
def StaticRejection(
    img_dir: Union[str, Path],
    cur_img_name: Union[str, Path],
    prev_img_name: Union[str, Path],
    keyframe_dir: Union[str, Path],
    method: str = "superglue",
    resize_to: List[int] = [-1],
    verbose: bool = False,
):
    timer = AverageTimer()

    img_dir = Path(img_dir)
    cur_img_name = Path(cur_img_name)
    prev_img_name = Path(prev_img_name)
    cur_img = img_dir / cur_img_name
    prev_img = img_dir / prev_img_name
    assert img_dir.is_dir(), f"Invalid image directory {img_dir}"
    # Check if the two images exists, otherwise skip.
    try:
        assert (
            cur_img.exists()
        ), f"Current image {cur_img_name} does not exist in image folder"
        assert (
            prev_img.exists()
        ), f"Previous image {prev_img_name} does not exist in image folder"
    except AssertionError as err:
        print("!! Frame truncated !!")
        return None

    if method == "superglue":
        im1 = cv2.imread(str(cur_img), flags=cv2.IMREAD_GRAYSCALE)
        im2 = cv2.imread(str(prev_img), flags=cv2.IMREAD_GRAYSCALE)
        timer.update("loaded imgs")

        if resize_to != [-1]:
            assert isinstance(
                resize_to, list
            ), "Invid input for resize_to parameter. It must be a list of integers with the new image dimensions"
            w_new, h_new = process_resize(im1.shape[1], im1.shape[0], resize=resize_to)
            if any([im1.shape[1] > w_new, im1.shape[0] > h_new]):
                im1 = cv2.resize(im1, (w_new, h_new))
                im2 = cv2.resize(im2, (w_new, h_new))
                if verbose:
                    logging.info(f"Images resized to ({w_new},{h_new})")
            timer.update("resizing")

        suerglue_cfg = {
            "weights": "outdoor",
            "keypoint_threshold": 0.01,
            "max_keypoints": 128,
            "match_threshold": 0.2,
            "force_cpu": False,
        }
        matcher = SuperGlueMatcher(suerglue_cfg)
        mkpts = matcher.match(np.asarray(im1), np.asarray(im2))
        timer.update("matching")
        timer.print(f"Static rejection {method}")

        # mkpts = matcher.geometric_verification(
        #     threshold=2,
        #     confidence=0.99,
        #     symmetric_error_check=False,
        # )
        # matcher.viz_matches("test.png")

    elif method == "loftr":
        device = torch.device("cuda")
        timg0, res_ratio0 = load_torch_image(
            cur_img, device=device, resize_to=resize_to, as_grayscale=True
        )
        timg1, res_ratio1 = load_torch_image(
            prev_img, device=device, resize_to=resize_to, as_grayscale=True
        )
        matcher = KF.LoFTR(pretrained="outdoor").to(device).eval()

        with torch.inference_mode():
            input_dict = {"image0": timg0, "image1": timg1}
            correspondences = matcher(input_dict)
    # print("Done.")


if __name__ == "__main__":
    img_dir = "imgs"
    keyframe_dir = "colmap_imgs"
    cur_img_name = "1403636579763555584.jpg"
    prev_img_name = "1403636580763555584.jpg"
    resize_to = [480]

    StaticRejection(
        img_dir,
        cur_img_name,
        prev_img_name,
        keyframe_dir=keyframe_dir,
        method="superglue",
        resize_to=resize_to,
    )

    print("Done")
