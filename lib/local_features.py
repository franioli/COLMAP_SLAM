from pathlib import Path
from typing import List, Tuple, Union

import cv2
import numpy as np

from lib.thirdparty.alike.alike import ALike, configs


class LocalFeatures:
    def __init__(
        self,
        method: str,
        n_features: int,
        cfg: dict = None,
    ) -> None:
        self.n_features = n_features
        self.method = method

        self.kpts = {}
        self.descriptors = {}

        # If method is ALIKE, load Alike model weights
        if self.method == "ALIKE":
            self.alike_cfg = cfg
            self.model = ALike(
                **configs[self.alike_cfg.model],
                device=self.alike_cfg.device,
                top_k=self.alike_cfg.top_k,
                scores_th=self.alike_cfg.scores_th,
                n_limit=self.alike_cfg.n_limit,
            )

    def ORB(self, images: List[Path]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        for im_path in images:
            im_path = Path(im_path)
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

            self.kpts[im_path.stem] = kpts
            self.descriptors[im_path.stem] = des

        return self.kpts, self.descriptors

    def ALIKE(self, images: List[Path]):
        for im_path in images:
            img = cv2.cvtColor(cv2.imread(str(im_path)), cv2.COLOR_BGR2RGB)
            features = self.model(img, sub_pixel=self.alike_cfg.subpixel)

            self.kpts[im_path.stem] = features["keypoints"]
            self.descriptors[im_path.stem] = features["descriptors"]

        return self.kpts, self.descriptors
