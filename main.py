# POSSIBLE CRITICAL POINTS:
# Check how input images are sorted >imgs = sorted(imgs, key=lambda x: int(x[6:-4]))
# Initialization of the photogrammetric model. If necessary the initial pair can be fixed in the lib/mapper_first_loop.ini file
# Exif gnss data works properly only when directions are N and E

# NOTES:
# Maybe newer_imgs can be eliminated
# matcher.ini us used only in the first loop. In the others, options are directly passed to the sequential_matcher API, see the code

# FEATURES TO BE ADDED:
# restart when the reconstruction breaks
# treat images with a class, so it will be easer to keep track og id, original, names, orientation, etc

import configparser
import os
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import piexif
from matplotlib import interactive
from mpl_toolkits import mplot3d
from pyquaternion import quaternion
from scipy import linalg
from scipy.spatial.transform import Rotation as R

from lib import (EKF, ConvertGnssRefSystm, ExtractCustomFeatures,
                 covariance_mat, database, export_cameras, static_rejection)
from lib.utils import Helmert, Id2name

### OPTIONS FOR EKF - Development temporarily interrupted, do not change values
T = 0.1
state_init = False

### MAIN STARTS HERE
# Working directories
CURRENT_DIR = Path(os.getcwd())
TEMP_DIR = CURRENT_DIR / "temp"
KEYFRAMES_DIR = CURRENT_DIR / "colmap_imgs"
OUT_FOLDER = CURRENT_DIR / "outs"
DATABASE = CURRENT_DIR / "outs" / "db.db"

# Import conf options
config = configparser.ConfigParser()
config.read(CURRENT_DIR / "config.ini", encoding="utf-8")

OS = config["DEFAULT"]["OS"]
if OS == "windows":
    colmap_exe = "COLMAP.bat"
elif OS == "linux":
    colmap_exe = "colmap"
else:
    print("OS in conf.ini must be windows or linux")
    quit()

# DEFAULT
USE_SERVER = config["DEFAULT"].getboolean("USE_SERVER")
LAUNCH_SERVER_PATH = Path(config["DEFAULT"]["LAUNCH_SERVER_PATH"])
DEBUG = config["DEFAULT"].getboolean("DEBUG")
MAX_IMG_BATCH_SIZE = int(config["DEFAULT"]["MAX_IMG_BATCH_SIZE"])
STATIC_IMG_REJECTION_METHOD = config["DEFAULT"][
    "STATIC_IMG_REJECTION_METHOD"
]  # 'radiometric' or 'root_sift'
SLEEP_TIME = float(config["DEFAULT"]["SLEEP_TIME"])
LOOP_CYCLES = int(config["DEFAULT"]["LOOP_CYCLES"])
COLMAP_EXE_PATH = Path(config["DEFAULT"]["COLMAP_EXE_PATH"])
IMGS_FROM_SERVER = Path(
    config["DEFAULT"]["IMGS_FROM_SERVER"]
)  # Path(r"/home/luca/Scrivania/3DOM/Github_lcmrl/Server_Connection/c++_send_images/imgs")
MAX_N_FEATURES = int(config["DEFAULT"]["MAX_N_FEATURES"])
INITIAL_SEQUENTIAL_OVERLAP = int(config["DEFAULT"]["INITIAL_SEQUENTIAL_OVERLAP"])
SEQUENTIAL_OVERLAP = INITIAL_SEQUENTIAL_OVERLAP
ONLY_SLAM = config["DEFAULT"].getboolean("ONLY_SLAM")
CUSTOM_FEATURES = config["DEFAULT"].getboolean("CUSTOM_FEATURES")
PATH_TO_LOCAL_FEATURES = Path(config["DEFAULT"]["PATH_TO_LOCAL_FEATURES"])
CUSTOM_DETECTOR = config["DEFAULT"]["CUSTOM_DETECTOR"]

# EXTERNAL_SENSORS
USE_EXTERNAL_CAM_COORD = config["EXTERNAL_SENSORS"].getboolean("USE_EXTERNAL_CAM_COORD")
CAMERA_COORDINATES_FILE = Path(config["EXTERNAL_SENSORS"]["CAMERA_COORDINATES_FILE"])

# INCREMENTAL_RECONSTRUCTION
MIN_KEYFRAME_FOR_INITIALIZATION = int(
    config["INCREMENTAL_RECONSTRUCTION"]["MIN_KEYFRAME_FOR_INITIALIZATION"]
)
LOOP_CLOSURE_DETECTION = config["INCREMENTAL_RECONSTRUCTION"].getboolean(
    "LOOP_CLOSURE_DETECTION"
)
VOCAB_TREE = config["INCREMENTAL_RECONSTRUCTION"]["VOCAB_TREE"]


# Initialize variables
position_dict = {}
img_dict = {}
ref_matches = []
processed_imgs = []
img_batch = []
img_batch_n = []
oriented_imgs_batch = []
pointer = 0
delta = 0
ended_first_colmap_loop = False
total_imgs = "000000"
Xslam = []
Yslam = []
Zslam = []
Xkf = []
Ykf = []
Zkf = []
one_time = False
reference_imgs_dict = {}

# Manage output folders
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)
    os.makedirs(TEMP_DIR / "pair")
    os.makedirs(KEYFRAMES_DIR)
    os.makedirs(OUT_FOLDER)
else:
    shutil.rmtree(TEMP_DIR)
    shutil.rmtree(KEYFRAMES_DIR)
    shutil.rmtree(OUT_FOLDER)
    os.makedirs(TEMP_DIR)
    os.makedirs(TEMP_DIR / "pair")
    os.makedirs(KEYFRAMES_DIR)
    os.makedirs(OUT_FOLDER)

if not os.path.exists(IMGS_FROM_SERVER):
    os.makedirs(IMGS_FROM_SERVER)
else:
    shutil.rmtree(IMGS_FROM_SERVER)
    os.makedirs(IMGS_FROM_SERVER)


# If the camera coordinates are known from other sensors than gnss,
# they can be stores in camera_coord_other_sensors dictionary and used
# to scale the photogrammetric model
camera_coord_other_sensors = {}
if USE_EXTERNAL_CAM_COORD == True:
    with open(CAMERA_COORDINATES_FILE, "r") as gt_file:
        lines = gt_file.readlines()
        for line in lines[2:]:
            id, x, y, z, _ = line.split(" ", 4)
            camera_coord_other_sensors[id] = (x, y, z)


### MAIN LOOP
# Stream of input data
p = subprocess.Popen(["python3", "./plot.py"])
if USE_SERVER == True:
    p = subprocess.Popen([LAUNCH_SERVER_PATH])
else:
    p = subprocess.Popen(["python3", "./simulator.py"])

for i in range(LOOP_CYCLES):
    imgs = os.listdir(IMGS_FROM_SERVER)
    imgs = sorted(imgs, key=lambda x: int(x[:-4]))
    newer_imgs = False
    processed = 0

    # Choose if keeping the pair
    if len(imgs) < 2:
        print("[{}] Not enough images. len(imgs) < 2".format(i))

    elif len(imgs) >= 2:
        for c, img in enumerate(imgs):
            # Decide if new images are valid to be added to the sequential matching
            # Only new images found in the target folder are processed.
            # No more than MAX_IMG_BATCH_SIZE imgs are processed.
            if (
                img not in processed_imgs
                and c >= 1
                and c != pointer
                and c > pointer + delta
                and processed < MAX_IMG_BATCH_SIZE
            ):
                img1 = imgs[pointer]
                img2 = imgs[c]
                start = time.time()
                (
                    ref_matches,
                    newer_imgs,
                    total_imgs,
                    img_dict,
                    img_batch,
                    pointer,
                ) = static_rejection.StaticRejection(
                    STATIC_IMG_REJECTION_METHOD,
                    img1,
                    img2,
                    IMGS_FROM_SERVER,
                    CURRENT_DIR,
                    KEYFRAMES_DIR,
                    COLMAP_EXE_PATH,
                    MAX_N_FEATURES,
                    ref_matches,
                    DEBUG,
                    newer_imgs,
                    total_imgs,
                    img_dict,
                    img_batch,
                    pointer,
                    colmap_exe,
                )  # pointer, delta,
                end = time.time()
                print("STATIC CHECK {}s".format(end - start), end="\r")
                processed_imgs.append(img)
                processed += 1

                # Load exif data and store GNSS position if present
                # or load camera cooridnates from other sensors
                exif_data = []
                try:
                    exif_data = piexif.load("{}/imgs/{}".format(os.getcwd(), img2))
                except:
                    print("Error loading exif data. Image file could be corrupted.")

                if exif_data != [] and len(exif_data["GPS"].keys()) != 0:
                    print("img2", img2)
                    lat = exif_data["GPS"][2]
                    long = exif_data["GPS"][4]
                    alt = exif_data["GPS"][6]

                    enuX, enuY, enuZ = ConvertGnssRefSystm.CovertGnssRefSystm(
                        lat, long, alt
                    )

                    position_dict[img2] = {
                        "GPSLatitude": exif_data["GPS"][2],
                        "GPSLongitude": exif_data["GPS"][4],
                        "GPSAltitude": exif_data["GPS"][6],
                        "enuX": enuX,
                        "enuY": enuY,
                        "enuZ": enuZ,
                        "slamX": "-",
                        "slamY": "-",
                        "slamZ": "-",
                    }

                elif exif_data != [] and img2 in camera_coord_other_sensors.keys():
                    print("img2", img2)
                    enuX, enuY, enuZ = (
                        camera_coord_other_sensors[img2][0],
                        camera_coord_other_sensors[img2][1],
                        camera_coord_other_sensors[img2][2],
                    )
                    position_dict[img2] = {
                        "enuX": enuX,
                        "enuY": enuY,
                        "enuZ": enuZ,
                        "slamX": "-",
                        "slamY": "-",
                        "slamZ": "-",
                    }

                else:
                    position_dict[img2] = {
                        "enuX": "-",
                        "enuY": "-",
                        "enuZ": "-",
                        "slamX": "-",
                        "slamY": "-",
                        "slamZ": "-",
                    }

    # INCREMENTAL RECONSTRUCTION
    kfrms = os.listdir(KEYFRAMES_DIR)
    kfrms.sort()

    if len(kfrms) >= MIN_KEYFRAME_FOR_INITIALIZATION and newer_imgs == True:
        start_loop = time.time()
        print(f"[LOOP : {i}]")
        print(f"DYNAMIC MATCHING WINDOW: ", SEQUENTIAL_OVERLAP)

        # FIRST LOOP IN COLMAP - INITIALIZATION
        if ended_first_colmap_loop == False:
            if CUSTOM_FEATURES == False:
                # Initialize an empty database
                st_time = time.time()
                if not os.path.exists(DATABASE):
                    subprocess.run(
                        [
                            COLMAP_EXE_PATH / f"{colmap_exe}",
                            "database_creator",
                            "--database_path",
                            DATABASE,
                        ],
                        stdout=subprocess.DEVNULL,
                    )
                end_time = time.time()
                print("DATABASE INITIALIZATION: ", end_time - st_time)

                # Feature extraction
                st_time = time.time()
                p = subprocess.run(
                    [
                        COLMAP_EXE_PATH / f"{colmap_exe}",
                        "feature_extractor",
                        "--project_path",
                        CURRENT_DIR / "lib" / "sift_first_loop.ini",
                    ],
                    stdout=subprocess.DEVNULL,
                )
                end_time = time.time()
                print("FEATURE EXTRACTION: ", end_time - st_time)

            elif CUSTOM_FEATURES == True:
                # Initialize an empty database
                st_time = time.time()
                if not os.path.exists(DATABASE):
                    subprocess.run(
                        [
                            COLMAP_EXE_PATH / f"{colmap_exe}",
                            "database_creator",
                            "--database_path",
                            DATABASE,
                        ],
                        stdout=subprocess.DEVNULL,
                    )
                end_time = time.time()
                print("DATABASE INITIALIZATION: ", end_time - st_time)

                # Feature extraction
                st_time = time.time()
                ExtractCustomFeatures.ExtractCustomFeatures(
                    CUSTOM_DETECTOR,
                    PATH_TO_LOCAL_FEATURES,
                    DATABASE,
                    kfrms,
                    img_dict,
                    KEYFRAMES_DIR,
                )
                end_time = time.time()
                print("FEATURE EXTRACTION: ", end_time - st_time)

            # Sequential matcher
            st_time = time.time()
            subprocess.run(
                [
                    COLMAP_EXE_PATH / f"{colmap_exe}",
                    "sequential_matcher",
                    "--project_path",
                    CURRENT_DIR / "lib" / "matcher.ini",
                ],
                stdout=subprocess.DEVNULL,
            )
            end_time = time.time()
            print("SEQUENTIAL MATCHER: ", end_time - st_time)

            # Triangulation and BA
            st_time = time.time()
            subprocess.run(
                [
                    COLMAP_EXE_PATH / f"{colmap_exe}",
                    "mapper",
                    "--project_path",
                    CURRENT_DIR / "lib" / "mapper_first_loop.ini",
                ],
                stdout=subprocess.DEVNULL,
            )
            end_time = time.time()
            print("MAPPER: ", end_time - st_time)

            # Convert model from binary to txt
            st_time = time.time()
            subprocess.run(
                [
                    COLMAP_EXE_PATH / f"{colmap_exe}",
                    "model_converter",
                    "--input_path",
                    OUT_FOLDER / "0",
                    "--output_path",
                    OUT_FOLDER,
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
            if CUSTOM_FEATURES == False:
                # Feature extraction
                st_time = time.time()
                subprocess.call(
                    [
                        COLMAP_EXE_PATH / f"{colmap_exe}",
                        "feature_extractor",
                        "--project_path",
                        CURRENT_DIR / "lib" / "sift.ini",
                    ],
                    stdout=subprocess.DEVNULL,
                )
                end_time = time.time()
                print("FEATURE EXTRACTION: ", end_time - st_time)

            elif CUSTOM_FEATURES == True:
                # Feature extraction
                st_time = time.time()
                ExtractCustomFeatures.ExtractCustomFeatures(
                    CUSTOM_DETECTOR,
                    PATH_TO_LOCAL_FEATURES,
                    DATABASE,
                    kfrms,
                    img_dict,
                    KEYFRAMES_DIR,
                )
                end_time = time.time()
                print("FEATURE EXTRACTION: ", end_time - st_time)

            # Sequential matcher
            st_time = time.time()
            # p = subprocess.Popen([COLMAP_EXE_PATH / "colmap", "sequential_matcher", "--project_path", CURRENT_DIR / "lib" / "matcher.ini"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
            if LOOP_CLOSURE_DETECTION == False:
                # Matching without loop closures (for all kind of local features)
                p = subprocess.call(
                    [
                        COLMAP_EXE_PATH / f"{colmap_exe}",
                        "sequential_matcher",
                        "--database_path",
                        DATABASE,
                        "--SequentialMatching.overlap",
                        "{}".format(SEQUENTIAL_OVERLAP),
                        "--SequentialMatching.quadratic_overlap",
                        "1",
                    ],
                    stdout=subprocess.DEVNULL,
                )
            elif LOOP_CLOSURE_DETECTION == True and CUSTOM_FEATURES == False:
                # Matching with loop closures (only for RootSIFT)
                p = subprocess.call(
                    [
                        COLMAP_EXE_PATH / f"{colmap_exe}",
                        "sequential_matcher",
                        "--database_path",
                        DATABASE,
                        "--SequentialMatching.overlap",
                        "{}".format(SEQUENTIAL_OVERLAP),
                        "--SequentialMatching.quadratic_overlap",
                        "1",
                        "--SequentialMatching.loop_detection",
                        "1",
                        "--SequentialMatching.vocab_tree_path",
                        VOCAB_TREE,
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
                    COLMAP_EXE_PATH / f"{colmap_exe}",
                    "mapper",
                    "--project_path",
                    CURRENT_DIR / "lib" / "mapper.ini",
                ],
                stdout=subprocess.DEVNULL,
            )
            end_time = time.time()
            print("MAPPER: ", end_time - st_time)

            # Convert model from binary to txt
            st_time = time.time()
            p = subprocess.call(
                [
                    COLMAP_EXE_PATH / f"{colmap_exe}",
                    "model_converter",
                    "--input_path",
                    OUT_FOLDER,
                    "--output_path",
                    OUT_FOLDER,
                    "--output_type",
                    "TXT",
                ],
                stdout=subprocess.DEVNULL,
            )
            end_time = time.time()
            print("MODEL CONVERSION: ", end_time - st_time)

        # Export cameras
        if DEBUG:
            lines, oriented_dict = export_cameras.ExportCameras(
                OUT_FOLDER / "images.txt", img_dict
            )
            with open(OUT_FOLDER / "loc.txt", "w") as file:
                for line in lines:
                    file.write(line)

        # Keep track of sucessfully oriented frames in the current img_batch
        for im_input_format in img_batch:
            im_zero_format = img_dict[im_input_format]
            img_batch_n.append(int(im_zero_format[:-4]))
            if int(im_zero_format[:-4]) in list(oriented_dict.keys()):
                oriented_imgs_batch.append(int(im_zero_format[:-4]))

        # Define new reference img (pointer)
        last_img_n = max(list(oriented_dict.keys()))
        max_img_n = max(img_batch_n)
        img_name = Id2name(last_img_n)
        inverted_img_dict = {v: k for k, v in img_dict.items()}
        for c, el in enumerate(imgs):
            if el == inverted_img_dict[img_name]:
                pointer = c  # pointer to the last oriented image
        delta = max_img_n - last_img_n

        # Update dynamic window for sequential matching
        if delta != 0:
            SEQUENTIAL_OVERLAP = INITIAL_SEQUENTIAL_OVERLAP + 2 * delta
        else:
            SEQUENTIAL_OVERLAP = INITIAL_SEQUENTIAL_OVERLAP

        oriented_dict_list = list(oriented_dict.keys())
        oriented_dict_list.sort()

        # Calculate transformation to report new slam solution on the reference one
        if one_time == True:
            list1 = []
            list2 = []
            for img_id in oriented_dict_list:
                img_name = inverted_img_dict[Id2name(img_id)]
                if img_name in reference_imgs_dict.keys():
                    list1.append(oriented_dict[img_id][1])
                    list2.append(reference_imgs_dict[img_name])
            R_, t_, scale_factor_ = Helmert(list1, list2, OS, DEBUG)

        # Apply rotantion matrix to move the updated photogrammetric model to the first model reference system
        for img_id in oriented_dict_list:
            img_name = inverted_img_dict[Id2name(img_id)]
            ref_img = list(position_dict.keys())[0]

            if img_name in position_dict.keys():
                if img_name == ref_img:
                    ref_img_id = img_id
                    quat1 = quaternion.Quaternion(oriented_dict[img_id][2][0])
                    t1 = oriented_dict[img_id][2][1]
                    position_dict[img_name]["slamX"] = 0.0
                    position_dict[img_name]["slamY"] = 0.0
                    position_dict[img_name]["slamZ"] = 0.0

                # The first model becomes the reference model (reference_imgs_dict)
                elif img_name != ref_img and one_time == False:
                    vec_pos = np.array(
                        [
                            [
                                oriented_dict[img_id][1][0],
                                oriented_dict[img_id][1][1],
                                oriented_dict[img_id][1][2],
                            ]
                        ]
                    ).T
                    camera_location = vec_pos
                    position_dict[img_name]["slamX"] = camera_location[0, 0]
                    position_dict[img_name]["slamY"] = camera_location[1, 0]
                    position_dict[img_name]["slamZ"] = camera_location[2, 0]
                    reference_imgs_dict[img_name] = (
                        camera_location[0, 0],
                        camera_location[1, 0],
                        camera_location[2, 0],
                    )

                # The subsequent models must be rotoranslated on the reference model, to always keep the same reference system
                elif img_name != ref_img and one_time == True:
                    vec_pos = np.array(
                        [
                            [
                                oriented_dict[img_id][1][0],
                                oriented_dict[img_id][1][1],
                                oriented_dict[img_id][1][2],
                            ]
                        ]
                    ).T
                    vec_pos_scaled = np.dot(R_, vec_pos) + t_
                    position_dict[img_name]["slamX"] = vec_pos_scaled[0, 0]
                    position_dict[img_name]["slamY"] = vec_pos_scaled[1, 0]
                    position_dict[img_name]["slamZ"] = vec_pos_scaled[2, 0]

        one_time = True

        if ONLY_SLAM != False:
            # INITIALIZATION SCALE FACTOR AND KALMAN FILTER
            if len(kfrms) == 30:
                # For images with both slam and gnss solution
                # georeference slam with Helmert transformation
                slam_coord = []
                gnss_coord = []
                for img in position_dict:
                    if position_dict[img]["enuX"] != "-":
                        gnss_coord.append(
                            (
                                position_dict[img]["enuX"],
                                position_dict[img]["enuY"],
                                position_dict[img]["enuZ"],
                            )
                        )
                        slam_coord.append(
                            (
                                position_dict[img]["slamX"],
                                position_dict[img]["slamY"],
                                position_dict[img]["slamZ"],
                            )
                        )
                # print(slam_coord, gnss_coord)

                R, t, scale_factor = Helmert(slam_coord, gnss_coord, OS, DEBUG)
                # print(R, t)

                # Store positions
                slam_coord = []
                for img in position_dict:
                    slam_coord.append(
                        (
                            position_dict[img]["slamX"],
                            position_dict[img]["slamY"],
                            position_dict[img]["slamZ"],
                        )
                    )
                for pos in slam_coord:
                    if pos[0] != "-":
                        pos = np.array([[pos[0]], [pos[1]], [pos[2]]])
                        scaled_pos = np.dot(R, pos) + t
                        Xslam.append(scaled_pos[0, 0])
                        Yslam.append(scaled_pos[1, 0])
                        Zslam.append(scaled_pos[2, 0])

                # plt.ion()
                # interactive(True)
                # fig = plt.figure()
                # ax = plt.axes(projection ='3d')
                # MIN = min([min(Xslam),min(Yslam),min(Zslam)])
                # MAX = max([max(Xslam),max(Yslam),max(Zslam)])
                # ax.cla()
                # ax.scatter(Xslam, Yslam, Zslam, 'black')
                # ax.set_title('c')
                ##ax.set_xticks([])
                ##ax.set_yticks([])
                ##ax.set_zticks([])
                # ax.view_init(azim=0, elev=90)
                # plt.show(block=True)
                # quit()

            elif len(kfrms) > 30:
                oriented_imgs_batch.sort()
                for img_id in oriented_imgs_batch:
                    # print(img_id)
                    img_name = inverted_img_dict[Id2name(img_id)]
                    # Positions in Sdr of the reference img
                    x = position_dict[img_name]["slamX"]
                    y = position_dict[img_name]["slamY"]
                    z = position_dict[img_name]["slamZ"]
                    observation = np.array([[x], [y], [z]])
                    scaled_observation = np.dot(R, observation) + t
                    Xslam.append(scaled_observation[0, 0])
                    Yslam.append(scaled_observation[1, 0])
                    Zslam.append(scaled_observation[2, 0])

                    if state_init == False:
                        X1 = position_dict[inverted_img_dict[Id2name(img_id - 2)]][
                            "slamX"
                        ]
                        Y1 = position_dict[inverted_img_dict[Id2name(img_id - 2)]][
                            "slamY"
                        ]
                        Z1 = position_dict[inverted_img_dict[Id2name(img_id - 2)]][
                            "slamZ"
                        ]
                        X2 = position_dict[inverted_img_dict[Id2name(img_id - 1)]][
                            "slamX"
                        ]
                        Y2 = position_dict[inverted_img_dict[Id2name(img_id - 1)]][
                            "slamY"
                        ]
                        Z2 = position_dict[inverted_img_dict[Id2name(img_id - 1)]][
                            "slamZ"
                        ]
                        X_1 = np.array([[X1, Y1, Z1]]).T
                        X_2 = np.array([[X2, Y2, Z2]]).T
                        X_1 = np.dot(R, X_1) + t
                        X_2 = np.dot(R, X_2) + t
                        V = (X_2 - X_1) / T
                        state_old = np.array(
                            [
                                [
                                    X_2[0, 0],
                                    X_2[1, 0],
                                    X_2[2, 0],
                                    V[0, 0],
                                    V[1, 0],
                                    V[2, 0],
                                    1,
                                ]
                            ]
                        ).T
                        state_init = True
                        P = covariance_mat.Pini()

                    # Smooth with EKF
                    # state_new, P_new, lambd = EKF.ExtendedKalmanFilter(state_old, P, covariance_mat.F(T), covariance_mat.Q(0.0009, 0.0001), scaled_observation, covariance_mat.R(0.1))
                    state_new, P_new, lambd = EKF.ExtendedKalmanFilter(
                        state_old,
                        P,
                        covariance_mat.F(T),
                        covariance_mat.Q(0.0001, 0.000001),
                        scaled_observation,
                        covariance_mat.R(0.01),
                    )

                    Xkf.append(state_old[0, 0])
                    Ykf.append(state_old[1, 0])
                    Zkf.append(state_old[2, 0])
                    state_old = state_new
                    P = P_new
                    # print("lambd", lambd)

                    plt.ion()
                    interactive(True)
                    fig = plt.figure()
                    ax = plt.axes(projection="3d")
                    MIN = min([min(Xslam), min(Yslam), min(Zslam)])
                    MAX = max([max(Xslam), max(Yslam), max(Zslam)])
                    ax.cla()
                    ax.scatter(Xslam, Yslam, Zslam, "black")
                    ax.scatter(Xkf, Ykf, Zkf, "red")
                    ax.set_title("c")
                    # ax.set_xticks([])
                    # ax.set_yticks([])
                    # ax.set_zticks([])
                    ax.view_init(azim=0, elev=90)
                    plt.show(block=True)

                    # predict new position with EKF (to calibrate scale factor so more accuracy on Q and less on R)
                    # if GNSS present
                    # Use the known prediction from slam and apply KF

                    # Print scale factor

        img_batch = []
        oriented_imgs_batch = []
        end_loop = time.time()
        print("LOOP TIME {}s\n".format(end_loop - start_loop))

    time.sleep(SLEEP_TIME)
