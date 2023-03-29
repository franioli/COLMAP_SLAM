from typing import List, Tuple, Union
from pathlib import Path
import numpy as np
import cv2

class LocalFeatures:
    def __init__(
        self,
        imgs: List[str],
        imgs_directory: Union[str, Path],
        n_features: int,
        method: str,
    ) -> None:
        self.imgs = imgs
        self.imgs_directory = imgs_directory
        self.n_features = n_features
        self.method = method

    def ORB(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        all_kpts = []
        all_descriptors = []

        for img in self.imgs:
            im_path = self.imgs_directory / img
            im = cv2.imread(str(im_path), cv2.IMREAD_GRAYSCALE)
            orb = cv2.ORB_create(nfeatures=self.n_features)
            kp = orb.detect(im, None)
            kp, des = orb.compute(im, kp)
            kpts = np.zeros((len(kp), 2))

            for j in range(len(kp)):
                kpts[j, 0], kpts[j, 1] = cv2.KeyPoint_convert(kp)[j][0], cv2.KeyPoint_convert(kp)[j][1]
            
            one_matrix = np.ones((len(kp), 1))
            kpts = np.append(kpts, one_matrix, axis=1)
            zero_matrix = np.zeros((len(kp), 1))
            kpts = np.append(kpts, zero_matrix, axis=1).astype(np.float32)

            zero_matrix = np.zeros((np.shape(des)[0], 96))
            des = np.append(des, zero_matrix, axis=1).astype(np.float32)
            des = np.absolute(des)
            des = des * 512 / np.linalg.norm(des, axis=1).reshape((-1, 1))
            des = np.round(des)
            des = np.array(des, dtype=np.uint8)

            all_kpts.append(kpts)
            all_descriptors.append(des)
        
        return all_kpts, all_descriptors

    def run(self):
        if self.method == 'ORB': self.ORB()
