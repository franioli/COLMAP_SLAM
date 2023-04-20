from pathlib import Path
from typing import List, Tuple, Union

import cv2
import numpy as np

from lib.thirdparty.alike.alike import ALike, configs


class LocalFeatures:
    def __init__(
        self,
        imgs: List[str],
        n_features: int,
        method: str,
        cfg: dict,
    ) -> None:
        self.imgs = imgs
        self.n_features = n_features
        self.method = method
        self.cfg = cfg

        # If method is ALIKE, load Alike model weights
        if self.method == "ALIKE":
            self.model = ALike(
                **configs[self.cfg.model],
                device=self.cfg.device,
                top_k=self.cfg.top_k,
                scores_th=self.cfg.scores_th,
                n_limit=self.cfg.n_limit,
            )

    def ORB(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        all_kpts = []
        all_descriptors = []

        for im_path in self.imgs:
            im = cv2.imread(str(im_path), cv2.IMREAD_GRAYSCALE)
            orb = cv2.ORB_create(nfeatures=self.n_features)
            kp = orb.detect(im, None)
            kp, des = orb.compute(im, kp)
            kpts = cv2.KeyPoint_convert(kp)

            one_matrix = np.ones((len(kp), 1))
            kpts = np.append(kpts, one_matrix, axis=1)
            zero_matrix = np.zeros((len(kp), 1))
            kpts = np.append(kpts, zero_matrix, axis=1).astype(np.float32)

            zero_matrix = np.zeros((des.shape[0], 96))
            des = np.append(des, zero_matrix, axis=1).astype(np.float32)
            des = np.absolute(des)
            des = des * 512 / np.linalg.norm(des, axis=1).reshape((-1, 1))
            des = np.round(des)
            des = np.array(des, dtype=np.uint8)

            all_kpts.append(kpts)
            all_descriptors.append(des)

        return all_kpts, all_descriptors

    def ALIKE(self, images: List[Path]):
        all_kpts = []
        all_descriptors = []
        for im_path in images:
            img = cv2.cvtColor(cv2.imread(str(self.im_path)), cv2.COLOR_BGR2RGB)
            features = self.model(img, sub_pixel=self.cfg.subpixel)

            all_kpts.append(features["keypoints"])
            all_descriptors.append(features["descriptors"])

        return all_kpts, all_descriptors

    def run(self):
        if self.method == "ORB":
            self.ORB()
