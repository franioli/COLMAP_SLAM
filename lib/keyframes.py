import numpy as np


class KeyFrame:
    def __init__(self, image_name, keyframe_id, keyframe_name, camera_id, image_id):
        self.image_name = image_name
        self.image_id = image_id
        self.keyframe_id = keyframe_id
        self.keyframe_name = keyframe_name
        self.camera_id = camera_id

        self.n_keypoints = 0
        self.oriented = False

        # Position
        self.GPSLatitude = "-"
        self.GPSLongitude = "-"
        self.GPSAltitude = "-"
        self.enuX = "-"
        self.enuY = "-"
        self.enuZ = "-"
        self.slamX = "-"
        self.slamY = "-"
        self.slamZ = "-"


class KeyFrameList:
    def __init__(self):
        self._keyframes = []
        self._current_idx = 0

    def __len__(self):
        return len(self._keyframes)

    def __getitem__(self, keyframe_id):
        return self._keyframes[keyframe_id]

    def __iter__(self):
        return self

    def __next__(self):
        if self._current_idx >= len(self._keyframes):
            raise StopIteration
        cur = self._current_idx
        self._current_idx += 1
        return self._keyframes[cur]

    def __repr__(self) -> str:
        return f"KeyframeList with {len(self._keyframes)} keyframes."

    @property
    def keyframes(self):
        return self._keyframes

    def add_keyframe(self, keyframe: KeyFrame):
        self._keyframes.append(keyframe)


if __name__ == "__main__":
    pass
