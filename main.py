# POSSIBLE CRITICAL POINTS:
# Initialization of the photogrammetric model. If necessary the initial pair can be fixed in the lib/mapper_first_loop.ini file
# Exif gnss data works properly only when directions are N and E

# NOTE:
# matcher.ini us used only in the first loop. In the others, options are directly passed to the sequential_matcher API, see the code
# keyframe_obj = list(filter(lambda obj: obj.image_name == img, keyframes_list))[0] Maybe is quicker to cycle with for loop

# TODO: FEATURES TO BE ADDED: restart when the reconstruction breaks

# NOTE: install easydict as additional dependancy

import configparser
import os
import shutil
import subprocess
import time
from pathlib import Path
import pickle
import cv2
import matplotlib.pyplot as plt
import numpy as np
import piexif
from easydict import EasyDict as edict

from matplotlib import interactive
from mpl_toolkits import mplot3d
from pyquaternion import quaternion
from scipy import linalg
from scipy.spatial.transform import Rotation as R

from lib import (
    EKF,
    ConvertGnssRefSystm,
    ExtractCustomFeatures,
    covariance_mat,
    database,
    export_cameras,
    static_rejection,
    keyframe_selection,
)
from lib.keyframes import KeyFrame, KeyFrameList
from lib.utils import Helmert, Id2name


### OPTIONS FOR EKF - Development temporarily interrupted, do not change values
T = 0.1
state_init = False

# Configuration file
CFG_FILE = "config.ini"


# Import conf options
# TODO: Move initialization class to new file
# TODO: centralized all options in config.ini file and cfg dictionary
# TODO: change parameter names to lowercase (both in Inizialization class and main)
# TODO: organize parameters in tree structure of dictionaries (or similar) to easily pass to function set of parameters
# TODO: add checks on parameters
class Inizialization:
    """Parse the configuration file and initialize the object's properties"""

    def __init__(self, cfg_file: str) -> None:
        """
        Args:
            cfg_file (str): Path to the configuration file
        """
        self.cfg_file = cfg_file

    def parse_config_file(self) -> edict:
        """
        Parse the configuration file and store the values in a dictionary

        Returns:
            dict: A dictionary containing all the configuration options
        """
        config = configparser.ConfigParser()
        config.read(self.cfg_file, encoding="utf-8")

        cfg = edict({})

        # DEFAULT
        cfg.OS = config["DEFAULT"]["OS"]
        cfg.USE_SERVER = config["DEFAULT"].getboolean("USE_SERVER")
        cfg.LAUNCH_SERVER_PATH = Path(config["DEFAULT"]["LAUNCH_SERVER_PATH"])
        cfg.DEBUG = config["DEFAULT"].getboolean("DEBUG")
        cfg.MAX_IMG_BATCH_SIZE = int(config["DEFAULT"]["MAX_IMG_BATCH_SIZE"])
        cfg.SLEEP_TIME = float(config["DEFAULT"]["SLEEP_TIME"])
        cfg.LOOP_CYCLES = int(config["DEFAULT"]["LOOP_CYCLES"])
        cfg.COLMAP_EXE_PATH = Path(config["DEFAULT"]["COLMAP_EXE_PATH"])
        cfg.IMGS_FROM_SERVER = Path(
            config["DEFAULT"]["IMGS_FROM_SERVER"]
        )  # Path(r"/home/luca/Scrivania/3DOM/Github_lcmrl/Server_Connection/c++_send_images/imgs")
        cfg.IMG_FORMAT = config["DEFAULT"]["IMG_FORMAT"]
        cfg.MAX_N_FEATURES = int(config["DEFAULT"]["MAX_N_FEATURES"])
        cfg.INITIAL_SEQUENTIAL_OVERLAP = int(
            config["DEFAULT"]["INITIAL_SEQUENTIAL_OVERLAP"]
        )
        cfg.SEQUENTIAL_OVERLAP = cfg.INITIAL_SEQUENTIAL_OVERLAP
        cfg.ONLY_SLAM = config["DEFAULT"].getboolean("ONLY_SLAM")
        cfg.CUSTOM_FEATURES = config["DEFAULT"].getboolean("CUSTOM_FEATURES")
        cfg.PATH_TO_LOCAL_FEATURES = Path(config["DEFAULT"]["PATH_TO_LOCAL_FEATURES"])
        cfg.CUSTOM_DETECTOR = config["DEFAULT"]["CUSTOM_DETECTOR"]

        # KEYFRAME_SELECTION
        cfg.KFS_METHOD = config["KEYFRAME_SELECTION"]["METHOD"]
        cfg.KFS_LOCAL_FEATURE = config["KEYFRAME_SELECTION"]["LOCAL_FEATURE"]
        cfg.KFS_N_FEATURES = int(config["KEYFRAME_SELECTION"]["N_FEATURES"])

        # EXTERNAL_SENSORS
        cfg.USE_EXTERNAL_CAM_COORD = config["EXTERNAL_SENSORS"].getboolean(
            "USE_EXTERNAL_CAM_COORD"
        )
        cfg.CAMERA_COORDINATES_FILE = Path(
            config["EXTERNAL_SENSORS"]["CAMERA_COORDINATES_FILE"]
        )

        # INCREMENTAL_RECONSTRUCTION
        cfg.MIN_KEYFRAME_FOR_INITIALIZATION = int(
            config["INCREMENTAL_RECONSTRUCTION"]["MIN_KEYFRAME_FOR_INITIALIZATION"]
        )
        cfg.LOOP_CLOSURE_DETECTION = config["INCREMENTAL_RECONSTRUCTION"].getboolean(
            "LOOP_CLOSURE_DETECTION"
        )
        cfg.VOCAB_TREE = config["INCREMENTAL_RECONSTRUCTION"]["VOCAB_TREE"]

        self.cfg = cfg

        return self.cfg

    def get_colmap_path(self) -> str:
        """
        Get the path to the COLMAP executable based on the operating system

        Returns:
            str: The path to the COLMAP executable
        """
        OS = self.cfg.OS
        assert OS in [
            "windows",
            "linux",
        ], "Invalid OS. OS in conf.ini must be windows or linux"
        if OS == "windows":
            self.colmap_exe = "COLMAP.bat"
        elif OS == "linux":
            self.colmap_exe = "colmap"
        return self.colmap_exe

    def set_working_directory(self):
        """
        set_working_directory _summary_
        """
        self.cfg.CURRENT_DIR = Path(os.getcwd())
        self.cfg.TEMP_DIR = self.cfg.CURRENT_DIR / "temp"
        self.cfg.KEYFRAMES_DIR = self.cfg.CURRENT_DIR / "colmap_imgs"
        self.cfg.OUT_FOLDER = self.cfg.CURRENT_DIR / "outs"
        self.cfg.DATABASE = self.cfg.CURRENT_DIR / "outs" / "db.db"

    def manage_output_folders(self) -> bool:
        """
        manage_output_folders _summary_

        Returns:
            bool: _description_
        """
        if not os.path.exists(self.cfg.TEMP_DIR):
            os.makedirs(self.cfg.TEMP_DIR)
            os.makedirs(self.cfg.TEMP_DIR / "pair")
            os.makedirs(self.cfg.KEYFRAMES_DIR)
            os.makedirs(self.cfg.OUT_FOLDER)
        else:
            shutil.rmtree(self.cfg.TEMP_DIR)
            shutil.rmtree(self.cfg.KEYFRAMES_DIR)
            shutil.rmtree(self.cfg.OUT_FOLDER)
            os.makedirs(self.cfg.TEMP_DIR)
            os.makedirs(self.cfg.TEMP_DIR / "pair")
            os.makedirs(self.cfg.KEYFRAMES_DIR)
            os.makedirs(self.cfg.OUT_FOLDER)

        if not os.path.exists(self.cfg.IMGS_FROM_SERVER):
            os.makedirs(self.cfg.IMGS_FROM_SERVER)
        else:
            shutil.rmtree(self.cfg.IMGS_FROM_SERVER)
            os.makedirs(self.cfg.IMGS_FROM_SERVER)

        if os.path.exists("./keyframes.pkl"):
            os.remove("./keyframes.pkl")

        return True

    def inizialize(self) -> edict:
        """
        inizialize _summary_

        Returns:
            edict: _description_
        """
        self.parse_config_file()
        self.set_working_directory()
        self.get_colmap_path()
        self.manage_output_folders()

        return self.cfg


# Inizialize COLMAP SLAM problem
init = Inizialization(CFG_FILE)
cfg = init.inizialize()


# Initialize variables
keyframes_list = KeyFrameList()
img_dict = (
    {}
)  ############################################# it is used in feature extraction, maybe it can be eliminated
processed_imgs = []
oriented_imgs_batch = []
pointer = 0  # pointer points to the last oriented image
delta = 0  # delta is equal to the number of processed but not oriented imgs
ended_first_colmap_loop = False
one_time = False  # It becomes true after the first batch of images is oriented
# The first batch of images define the reference system.
# At following epochs the photogrammetric model will be reported in this ref system.
reference_imgs = []


# If the camera coordinates are known from other sensors than gnss,
# they can be stores in camera_coord_other_sensors dictionary and used
# to scale the photogrammetric model
camera_coord_other_sensors = {}
if cfg.USE_EXTERNAL_CAM_COORD == True:
    with open(cfg.CAMERA_COORDINATES_FILE, "r") as gt_file:
        lines = gt_file.readlines()
        for line in lines[2:]:
            id, x, y, z, _ = line.split(" ", 4)
            camera_coord_other_sensors[id] = (x, y, z)

# Stream of input data
p = subprocess.Popen(["python3", "./plot.py"])
if cfg.USE_SERVER == True:
    p = subprocess.Popen([cfg.LAUNCH_SERVER_PATH])
else:
    p = subprocess.Popen(["python3", "./simulator.py"])


### MAIN LOOP
for i in range(cfg.LOOP_CYCLES):
    # Get sorted image list available in imgs folders
    imgs = sorted(cfg.IMGS_FROM_SERVER.glob(f"*.{cfg.IMG_FORMAT}"))
    img_batch = []

    newer_imgs = False  # To control that new keyframes are added
    processed = 0  # Number of processed images

    # Keyframe selection
    if len(imgs) < 2:
        continue

    elif len(imgs) >= 2:
        for c, img in enumerate(imgs):
            # Decide if new images are valid to be added to the sequential matching
            # Only new images found in the target folder are processed.
            # No more than MAX_IMG_BATCH_SIZE imgs are processed.
            if img in processed_imgs or c < 1 or processed >= cfg.MAX_IMG_BATCH_SIZE:
                continue

            print()
            print()
            print("pointer", pointer, "c", c)
            img1 = imgs[pointer]
            img2 = img
            start = time.time()
            old_n_keyframes = len(os.listdir(cfg.KEYFRAMES_DIR))

            (
                keyframes_list,
                pointer,
                delta,
            ) = keyframe_selection.KeyframeSelection(
                cfg.KFS_METHOD,
                cfg.KFS_LOCAL_FEATURE,
                cfg.KFS_N_FEATURES,
                img1,
                img2,
                cfg.KEYFRAMES_DIR,
                keyframes_list,
                pointer,
                delta,
            )

            # Set if new keyframes are added
            new_n_keyframes = len(os.listdir(cfg.KEYFRAMES_DIR))
            if new_n_keyframes - old_n_keyframes > 0:
                newer_imgs = True
                img_batch.append(img)
                keyframe_obj = list(
                    filter(lambda obj: obj.image_name == img, keyframes_list)
                )[0]

                # Load exif data and store GNSS position if present
                # or load camera cooridnates from other sensors
                exif_data = []
                try:
                    exif_data = piexif.load("{}/imgs/{}".format(os.getcwd(), img2))
                except:
                    print("Error loading exif data. Image file could be corrupted.")

                if exif_data != [] and len(exif_data["GPS"].keys()) != 0:
                    lat = exif_data["GPS"][2]
                    long = exif_data["GPS"][4]
                    alt = exif_data["GPS"][6]
                    enuX, enuY, enuZ = ConvertGnssRefSystm.CovertGnssRefSystm(
                        lat, long, alt
                    )
                    keyframe_obj.GPSLatitude = lat
                    keyframe_obj.GPSLongitude = long
                    keyframe_obj.GPSAltitude = alt
                    keyframe_obj.enuX = enuX
                    keyframe_obj.enuY = enuY
                    keyframe_obj.enuZ = enuZ

                elif exif_data != [] and img2 in camera_coord_other_sensors.keys():
                    print("img2", img2)
                    enuX, enuY, enuZ = (
                        camera_coord_other_sensors[img2][0],
                        camera_coord_other_sensors[img2][1],
                        camera_coord_other_sensors[img2][2],
                    )
                    keyframe_obj.enuX = enuX
                    keyframe_obj.enuY = enuY
                    keyframe_obj.enuZ = enuZ

            processed_imgs.append(img)
            processed += 1
            end = time.time()
            print("STATIC CHECK {}s".format(end - start), end="\r")

    # INCREMENTAL RECONSTRUCTION
    kfrms = os.listdir(cfg.KEYFRAMES_DIR)
    kfrms.sort()

    if len(kfrms) >= cfg.MIN_KEYFRAME_FOR_INITIALIZATION and newer_imgs == True:
        start_loop = time.time()
        print()
        print(f"[LOOP : {i}]")
        print(f"DYNAMIC MATCHING WINDOW: ", SEQUENTIAL_OVERLAP)
        print()

        # FIRST LOOP IN COLMAP - INITIALIZATION
        if ended_first_colmap_loop == False:
            if cfg.CUSTOM_FEATURES == False:
                # Initialize an empty database
                st_time = time.time()
                if not os.path.exists(cfg.DATABASE):
                    subprocess.run(
                        [
                            cfg.COLMAP_EXE_PATH / f"{cfg.colmap_exe}",
                            "database_creator",
                            "--database_path",
                            cfg.DATABASE,
                        ],
                        stdout=subprocess.DEVNULL,
                    )
                end_time = time.time()
                print("DATABASE INITIALIZATION: ", end_time - st_time)

                # Feature extraction
                st_time = time.time()
                p = subprocess.run(
                    [
                        cfg.COLMAP_EXE_PATH / f"{cfg.colmap_exe}",
                        "feature_extractor",
                        "--project_path",
                        cfg.CURRENT_DIR / "lib" / "sift_first_loop.ini",
                    ],
                    stdout=subprocess.DEVNULL,
                )
                end_time = time.time()
                print("FEATURE EXTRACTION: ", end_time - st_time)

            elif cfg.CUSTOM_FEATURES == True:
                # Initialize an empty database
                st_time = time.time()
                if not os.path.exists(cfg.DATABASE):
                    subprocess.run(
                        [
                            cfg.COLMAP_EXE_PATH / f"{cfg.colmap_exe}",
                            "database_creator",
                            "--database_path",
                            cfg.DATABASE,
                        ],
                        stdout=subprocess.DEVNULL,
                    )
                end_time = time.time()
                print("DATABASE INITIALIZATION: ", end_time - st_time)

                # Feature extraction
                st_time = time.time()
                ExtractCustomFeatures.ExtractCustomFeatures(
                    cfg.CUSTOM_DETECTOR,
                    cfg.PATH_TO_LOCAL_FEATURES,
                    cfg.DATABASE,
                    cfg.KEYFRAMES_DIR,
                    keyframes_list,
                )
                end_time = time.time()
                print("FEATURE EXTRACTION: ", end_time - st_time)

            # Sequential matcher
            st_time = time.time()
            subprocess.run(
                [
                    cfg.COLMAP_EXE_PATH / f"{cfg.colmap_exe}",
                    "sequential_matcher",
                    "--project_path",
                    cfg.CURRENT_DIR / "lib" / "matcher.ini",
                ],
                stdout=subprocess.DEVNULL,
            )
            end_time = time.time()
            print("SEQUENTIAL MATCHER: ", end_time - st_time)

            # Triangulation and BA
            st_time = time.time()
            subprocess.run(
                [
                    cfg.COLMAP_EXE_PATH / f"{cfg.colmap_exe}",
                    "mapper",
                    "--project_path",
                    cfg.CURRENT_DIR / "lib" / "mapper_first_loop.ini",
                ],
                stdout=subprocess.DEVNULL,
            )
            end_time = time.time()
            print("MAPPER: ", end_time - st_time)

            # Convert model from binary to txt
            st_time = time.time()
            subprocess.run(
                [
                    cfg.COLMAP_EXE_PATH / f"{cfg.colmap_exe}",
                    "model_converter",
                    "--input_path",
                    cfg.OUT_FOLDER / "0",
                    "--output_path",
                    cfg.OUT_FOLDER,
                    "--output_type",
                    "TXT",
                ],
                stdout=subprocess.DEVNULL,
            )
            end_time = time.time()
            print("MODEL CONVERSION: ", end_time - st_time)

            ended_first_colmap_loop = True

        # ALL COLMAP LOOPS AFTER THE FIRST - MODEL GROWTH
        elif ended_first_colmap_loop == True:
            if cfg.CUSTOM_FEATURES == False:
                # Feature extraction
                st_time = time.time()
                subprocess.call(
                    [
                        cfg.COLMAP_EXE_PATH / f"{cfg.colmap_exe}",
                        "feature_extractor",
                        "--project_path",
                        cfg.CURRENT_DIR / "lib" / "sift.ini",
                    ],
                    stdout=subprocess.DEVNULL,
                )
                end_time = time.time()
                print("FEATURE EXTRACTION: ", end_time - st_time)

            elif cfg.CUSTOM_FEATURES == True:
                # Feature extraction
                st_time = time.time()
                ExtractCustomFeatures.ExtractCustomFeatures(
                    cfg.CUSTOM_DETECTOR,
                    cfg.PATH_TO_LOCAL_FEATURES,
                    cfg.DATABASE,
                    cfg.KEYFRAMES_DIR,
                    keyframes_list,
                )
                end_time = time.time()
                print("FEATURE EXTRACTION: ", end_time - st_time)

            # Sequential matcher
            st_time = time.time()
            # p = subprocess.Popen([COLMAP_EXE_PATH / "colmap", "sequential_matcher", "--project_path", CURRENT_DIR / "lib" / "matcher.ini"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
            if cfg.LOOP_CLOSURE_DETECTION == False:
                # Matching without loop closures (for all kind of local features)
                p = subprocess.call(
                    [
                        cfg.COLMAP_EXE_PATH / f"{cfg.colmap_exe}",
                        "sequential_matcher",
                        "--database_path",
                        cfg.DATABASE,
                        "--SequentialMatching.overlap",
                        "{}".format(SEQUENTIAL_OVERLAP),
                        "--SequentialMatching.quadratic_overlap",
                        "1",
                    ],
                    stdout=subprocess.DEVNULL,
                )
            elif cfg.LOOP_CLOSURE_DETECTION == True and cfg.CUSTOM_FEATURES == False:
                # Matching with loop closures (only for RootSIFT)
                p = subprocess.call(
                    [
                        cfg.COLMAP_EXE_PATH / f"{cfg.colmap_exe}",
                        "sequential_matcher",
                        "--database_path",
                        cfg.DATABASE,
                        "--SequentialMatching.overlap",
                        "{}".format(SEQUENTIAL_OVERLAP),
                        "--SequentialMatching.quadratic_overlap",
                        "1",
                        "--SequentialMatching.loop_detection",
                        "1",
                        "--SequentialMatching.vocab_tree_path",
                        cfg.VOCAB_TREE,
                    ],
                    stdout=subprocess.DEVNULL,
                )
            else:
                print("Not compatible option for loop closure detection. Quit.")
                quit()
            end_time = time.time()
            print("SEQUENTIAL MATCHER: ", end_time - st_time)

            # Triangulation and BA
            st_time = time.time()
            subprocess.call(
                [
                    cfg.COLMAP_EXE_PATH / f"{cfg.colmap_exe}",
                    "mapper",
                    "--project_path",
                    cfg.CURRENT_DIR / "lib" / "mapper.ini",
                ],
                stdout=subprocess.DEVNULL,
            )
            end_time = time.time()
            print("MAPPER: ", end_time - st_time)

            # Convert model from binary to txt
            st_time = time.time()
            p = subprocess.call(
                [
                    cfg.COLMAP_EXE_PATH / f"{cfg.colmap_exe}",
                    "model_converter",
                    "--input_path",
                    cfg.OUT_FOLDER,
                    "--output_path",
                    cfg.OUT_FOLDER,
                    "--output_type",
                    "TXT",
                ],
                stdout=subprocess.DEVNULL,
            )
            end_time = time.time()
            print("MODEL CONVERSION: ", end_time - st_time)

        # Export cameras
        lines, oriented_dict = export_cameras.ExportCameras(
            cfg.OUT_FOLDER / "images.txt", keyframes_list
        )
        if cfg.DEBUG:
            with open(cfg.OUT_FOLDER / "loc.txt", "w") as file:
                for line in lines:
                    file.write(line)

        # Keep track of sucessfully oriented frames in the current img_batch
        for image in img_batch:
            keyframe_obj = list(
                filter(lambda obj: obj.image_name == image, keyframes_list)
            )[0]
            if keyframe_obj.keyframe_id in list(oriented_dict.keys()):
                oriented_imgs_batch.append(image)
                keyframe_obj.oriented = True

        # Define new reference img (pointer)
        print("list(oriented_dict.keys())")
        print(list(oriented_dict.keys()))
        last_oriented_keyframe = np.max(list(oriented_dict.keys()))
        print("last_oriented_keyframe", last_oriented_keyframe)
        keyframe_obj = list(
            filter(
                lambda obj: obj.keyframe_id == last_oriented_keyframe, keyframes_list
            )
        )[0]
        n_keyframes = len(os.listdir(cfg.KEYFRAMES_DIR))
        last_keyframe = list(
            filter(lambda obj: obj.keyframe_id == n_keyframes - 1, keyframes_list)
        )[0]
        last_keyframe_img_id = last_keyframe.image_id
        print("last_keyframe_img_id", last_keyframe_img_id)
        print("n_keyframes", n_keyframes)
        pointer = keyframe_obj.image_id  # pointer to the last oriented image
        print("pointer", pointer)
        delta = last_keyframe_img_id - pointer
        print("delta", delta)
        print()

        # Update dynamic window for sequential matching
        if delta != 0:
            SEQUENTIAL_OVERLAP = cfg.INITIAL_SEQUENTIAL_OVERLAP + 2 * (
                n_keyframes - last_oriented_keyframe
            )
        else:
            SEQUENTIAL_OVERLAP = cfg.INITIAL_SEQUENTIAL_OVERLAP

        ### STORE SLAM SOLUTION
        oriented_dict_list = list(oriented_dict.keys())
        oriented_dict_list.sort()

        # Define a reference img. All other oriented keyframes will be reported in the ref of the first keyframe
        if one_time == False:
            ref_img_id = oriented_dict_list[0]
            keyframe_obj = list(
                filter(lambda obj: obj.keyframe_id == ref_img_id, keyframes_list)
            )[0]
            ref_img_name = keyframe_obj.image_name
            reference_imgs.append(ref_img_name)
            keyframe_obj.slamX = 0.0
            keyframe_obj.slamY = 0.0
            keyframe_obj.slamZ = 0.0

        # Calculate transformation to report new slam solution on the reference one
        if one_time == True:
            list1 = []
            list2 = []
            for img_name in reference_imgs:
                # keyframe_obj = list(filter(lambda obj: obj.image_name == img_name, keyframes_list))[0]
                keyframe_obj = [
                    obj for obj in keyframes_list if obj.image_name == img_name
                ][0]
                img_id = keyframe_obj.keyframe_id
                if img_id in oriented_dict_list:
                    list1.append(oriented_dict[img_id][1])
                    list2.append(
                        (keyframe_obj.slamX, keyframe_obj.slamY, keyframe_obj.slamZ)
                    )
            R_, t_, scale_factor_ = Helmert(list1, list2, cfg.OS, cfg.DEBUG)

        # Apply rotantion matrix to move the updated photogrammetric model to the first model reference system
        for keyframe_id in oriented_dict_list:
            keyframe_obj = list(
                filter(lambda obj: obj.keyframe_id == keyframe_id, keyframes_list)
            )[0]
            # keyframe_obj = [obj for obj in keyframes_list if obj.keyframe_id == keyframe_id][0]
            img_name = keyframe_obj.image_name

            if img_name == ref_img_name:
                pass

            # The first model becomes the reference model (reference_imgs_dict)
            elif img_name != ref_img_name and one_time == False:
                camera_location = np.array(oriented_dict[keyframe_id][1]).reshape(
                    (3, 1)
                )
                keyframe_obj.slamX = camera_location[0, 0]
                keyframe_obj.slamY = camera_location[1, 0]
                keyframe_obj.slamZ = camera_location[2, 0]
                reference_imgs.append(img_name)

            # The subsequent models must be rotoranslated on the reference model, to always keep the same reference system
            elif img_name != ref_img_name and one_time == True:
                camera_location = np.array(oriented_dict[keyframe_id][1]).reshape(
                    (3, 1)
                )
                vec_pos_scaled = np.dot(R_, camera_location) + t_
                keyframe_obj.slamX = vec_pos_scaled[0, 0]
                keyframe_obj.slamY = vec_pos_scaled[1, 0]
                keyframe_obj.slamZ = vec_pos_scaled[2, 0]

        one_time = True
        with open("./keyframes.pkl", "wb") as f:
            pickle.dump(keyframes_list, f)

        ## TEMPORANELY NOT DEVELOPED. KEEP ONLY_SLAM == True
        # if ONLY_SLAM == False:
        #    # INITIALIZATION SCALE FACTOR AND KALMAN FILTER
        #    if len(kfrms) == 30:
        #        # For images with both slam and gnss solution
        #        # georeference slam with Helmert transformation
        #        slam_coord = []
        #        gnss_coord = []
        #        for img in position_dict:
        #            if position_dict[img]["enuX"] != "-":
        #                gnss_coord.append(
        #                    (
        #                        position_dict[img]["enuX"],
        #                        position_dict[img]["enuY"],
        #                        position_dict[img]["enuZ"],
        #                    )
        #                )
        #                slam_coord.append(
        #                    (
        #                        position_dict[img]["slamX"],
        #                        position_dict[img]["slamY"],
        #                        position_dict[img]["slamZ"],
        #                    )
        #                )
        #        # print(slam_coord, gnss_coord)
        #
        #        R, t, scale_factor = Helmert(slam_coord, gnss_coord, OS, DEBUG)
        #        # print(R, t)
        #
        #        # Store positions
        #        slam_coord = []
        #        for img in position_dict:
        #            slam_coord.append(
        #                (
        #                    position_dict[img]["slamX"],
        #                    position_dict[img]["slamY"],
        #                    position_dict[img]["slamZ"],
        #                )
        #            )
        #        for pos in slam_coord:
        #            if pos[0] != "-":
        #                pos = np.array([[pos[0]], [pos[1]], [pos[2]]])
        #                scaled_pos = np.dot(R, pos) + t
        #                Xslam.append(scaled_pos[0, 0])
        #                Yslam.append(scaled_pos[1, 0])
        #                Zslam.append(scaled_pos[2, 0])
        #
        #        # plt.ion()
        #        # interactive(True)
        #        # fig = plt.figure()
        #        # ax = plt.axes(projection ='3d')
        #        # MIN = min([min(Xslam),min(Yslam),min(Zslam)])
        #        # MAX = max([max(Xslam),max(Yslam),max(Zslam)])
        #        # ax.cla()
        #        # ax.scatter(Xslam, Yslam, Zslam, 'black')
        #        # ax.set_title('c')
        #        ##ax.set_xticks([])
        #        ##ax.set_yticks([])
        #        ##ax.set_zticks([])
        #        # ax.view_init(azim=0, elev=90)
        #        # plt.show(block=True)
        #        # quit()
        #
        #    elif len(kfrms) > 30:
        #        oriented_imgs_batch.sort()
        #        for img_id in oriented_imgs_batch:
        #            # print(img_id)
        #            img_name = inverted_img_dict[Id2name(img_id)]
        #            # Positions in Sdr of the reference img
        #            x = position_dict[img_name]["slamX"]
        #            y = position_dict[img_name]["slamY"]
        #            z = position_dict[img_name]["slamZ"]
        #            observation = np.array([[x], [y], [z]])
        #            scaled_observation = np.dot(R, observation) + t
        #            Xslam.append(scaled_observation[0, 0])
        #            Yslam.append(scaled_observation[1, 0])
        #            Zslam.append(scaled_observation[2, 0])
        #
        #            if state_init == False:
        #                X1 = position_dict[inverted_img_dict[Id2name(img_id - 2)]][
        #                    "slamX"
        #                ]
        #                Y1 = position_dict[inverted_img_dict[Id2name(img_id - 2)]][
        #                    "slamY"
        #                ]
        #                Z1 = position_dict[inverted_img_dict[Id2name(img_id - 2)]][
        #                    "slamZ"
        #                ]
        #                X2 = position_dict[inverted_img_dict[Id2name(img_id - 1)]][
        #                    "slamX"
        #                ]
        #                Y2 = position_dict[inverted_img_dict[Id2name(img_id - 1)]][
        #                    "slamY"
        #                ]
        #                Z2 = position_dict[inverted_img_dict[Id2name(img_id - 1)]][
        #                    "slamZ"
        #                ]
        #                X_1 = np.array([[X1, Y1, Z1]]).T
        #                X_2 = np.array([[X2, Y2, Z2]]).T
        #                X_1 = np.dot(R, X_1) + t
        #                X_2 = np.dot(R, X_2) + t
        #                V = (X_2 - X_1) / T
        #                state_old = np.array(
        #                    [
        #                        [
        #                            X_2[0, 0],
        #                            X_2[1, 0],
        #                            X_2[2, 0],
        #                            V[0, 0],
        #                            V[1, 0],
        #                            V[2, 0],
        #                            1,
        #                        ]
        #                    ]
        #                ).T
        #                state_init = True
        #                P = covariance_mat.Pini()
        #
        #            # Smooth with EKF
        #            # state_new, P_new, lambd = EKF.ExtendedKalmanFilter(state_old, P, covariance_mat.F(T), covariance_mat.Q(0.0009, 0.0001), scaled_observation, covariance_mat.R(0.1))
        #            state_new, P_new, lambd = EKF.ExtendedKalmanFilter(
        #                state_old,
        #                P,
        #                covariance_mat.F(T),
        #                covariance_mat.Q(0.0001, 0.000001),
        #                scaled_observation,
        #                covariance_mat.R(0.01),
        #            )
        #
        #            Xkf.append(state_old[0, 0])
        #            Ykf.append(state_old[1, 0])
        #            Zkf.append(state_old[2, 0])
        #            state_old = state_new
        #            P = P_new
        #            # print("lambd", lambd)
        #
        #            plt.ion()
        #            interactive(True)
        #            fig = plt.figure()
        #            ax = plt.axes(projection="3d")
        #            MIN = min([min(Xslam), min(Yslam), min(Zslam)])
        #            MAX = max([max(Xslam), max(Yslam), max(Zslam)])
        #            ax.cla()
        #            ax.scatter(Xslam, Yslam, Zslam, "black")
        #            ax.scatter(Xkf, Ykf, Zkf, "red")
        #            ax.set_title("c")
        #            # ax.set_xticks([])
        #            # ax.set_yticks([])
        #            # ax.set_zticks([])
        #            ax.view_init(azim=0, elev=90)
        #            plt.show(block=True)
        #
        #            # predict new position with EKF (to calibrate scale factor so more accuracy on Q and less on R)
        #            # if GNSS present
        #            # Use the known prediction from slam and apply KF
        #
        #            # Print scale factor

        img_batch = []
        oriented_imgs_batch = []
        end_loop = time.time()
        print("LOOP TIME {}s\n".format(end_loop - start_loop))

    time.sleep(cfg.SLEEP_TIME)
