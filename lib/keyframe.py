import numpy as np

class Keyframe:
    def __init__(self, image_name, keyframe_id, keyframe_name, camera_id, image_id):
        self.image_name = image_name
        self.image_id = image_id
        self.keyframe_id = keyframe_id
        self.keyframe_name = keyframe_name
        self.camera_id = camera_id

        self.n_keypoints = 0
        self.oriented = False

        # Position
        self.GPSLatitude = '-'
        self.GPSLongitude = '-'
        self.GPSAltitude = '-'
        self.enuX = '-'
        self.enuY = '-'
        self.enuZ = '-'
        self.slamX = '-'
        self.slamY = '-'
        self.slamZ = '-'
    
                    