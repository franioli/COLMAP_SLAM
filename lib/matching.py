import numpy as np
import cv2
from copy import deepcopy


def make_match_plot(
    img: np.ndarray, mpts1: np.ndarray, mpts2: np.ndarray
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
    match_img = deepcopy(img)
    for pt1, pt2 in zip(mpts1, mpts2):
        p1 = (int(round(pt1[0])), int(round(pt1[1])))
        p2 = (int(round(pt2[0])), int(round(pt2[1])))
        cv2.line(match_img, p1, p2, (0, 255, 0), lineType=16)
        cv2.circle(match_img, p2, 1, (0, 0, 255), -1, lineType=16)

    return match_img


class Matcher:
    def __init__(
        self,
        desc1: np.ndarray,
        desc2: np.ndarray,
    ):
        self.desc1 = desc1 / np.linalg.norm(desc1, axis=1, keepdims=True)
        self.desc2 = desc2 / np.linalg.norm(desc2, axis=1, keepdims=True)

    def mnn_matcher_cosine(self) -> np.ndarray:
        """
        Computes the nearest neighbor matches between two sets of descriptors.
        Args:
            desc1 (numpy.ndarray): First set of descriptors.
            desc2 (numpy.ndarray): Second set of descriptors.
        Returns:
            numpy.ndarray: An array of indices indicating the nearest neighbor matches between the two sets of descriptors.
        """
        sim = self.desc1 @ self.desc2.transpose()
        sim[sim < 0.8] = 0
        nn12 = np.argmax(sim, axis=1)
        nn21 = np.argmax(sim, axis=0)
        ids1 = np.arange(0, sim.shape[0])
        mask = ids1 == nn21[nn12]
        matches = np.stack([ids1[mask], nn12[mask]])
        # matches = np.stack([ids1, nn12])

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

        return make_match_plot(img, mpts1, mpts2)
