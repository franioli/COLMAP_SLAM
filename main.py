# POSSIBLE CRITICAL POINTS:
# Initialization of the photogrammetric model. If necessary the initial pair can be fixed in the lib/mapper_first_loop.ini file
# Exif gnss data works properly only when directions are N and E

# NOTE:
# matcher.ini us used only in the first loop. In the others, options are directly passed to the sequential_matcher API, see the code
# keyframe_obj = list(filter(lambda obj: obj.image_name == img, keyframes_list))[0] Maybe is quicker to cycle with for loop

# TODO: FEATURES TO BE ADDED: restart when the reconstruction breaks

# NOTE: install easydict as additional dependancy

import configparser
import logging
import os
import pickle
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import numpy as np
import piexif
from easydict import EasyDict as edict
from matplotlib import interactive
from matplotlib.animation import FuncAnimation
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
    utils,
)
from lib.keyframe_selection import KeyFrameSelector
from lib.keyframes import KeyFrame, KeyFrameList


def colmap_process() -> bool:
    pass


# def update_plot(ax):
#     if not os.path.exists("./keyframes.pkl"):
#         return

#     X = []
#     Y = []
#     Z = []
#     with open("./keyframes.pkl", "rb") as f:
#         my_list = pickle.load(f)
#         for obj in my_list:
#             if obj.slamX != "-":
#                 X.append(float(obj.slamX))
#             if obj.slamY != "-":
#                 Y.append(float(obj.slamY))
#             if obj.slamZ != "-":
#                 Z.append(float(obj.slamZ))

#     ax.cla()
#     ax.scatter(X, Y, Z, "black")
#     ax.set_title("c")
#     ax.set_xticks([])
#     ax.set_yticks([])
#     ax.set_zticks([])  # ax[2].set_zticks(np.arange(MIN, MAX, (MAX-MIN)/10))
#     ax.view_init(azim=0, elev=90)

#     plt.show(block=False)

#     Total_imgs = len(os.listdir("./colmap_imgs"))
#     N_oriented_cameras = len(X)
#     logging.info("Total: {}  Oriented: {}".format(Total_imgs, N_oriented_cameras))

import plotly.graph_objs as go
from plotly.subplots import make_subplots


def read_camera_eo(fname: str = "./keyframes.pkl") -> tuple:
    if not os.path.exists(fname):
        return

    X = []
    Y = []
    Z = []
    with open(fname, "rb") as f:
        my_list = pickle.load(f)
        for obj in my_list:
            if obj.slamX != "-":
                X.append(float(obj.slamX))
            if obj.slamY != "-":
                Y.append(float(obj.slamY))
            if obj.slamZ != "-":
                Z.append(float(obj.slamZ))

    return X, Y, Z


def create_plot():
    global camera_pos_trace, camera_data_trace, fig
    fig = make_subplots(rows=1, cols=1, specs=[[{"type": "scatter3d"}]])
    fig.update_layout(
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        title_text="Camera Trajectory",
    )

    # Add the camera position trace to the figure
    camera_pos_trace = go.Scatter3d(
        x=[0], y=[0], z=[0], mode="markers", marker=dict(size=10, color="red")
    )
    fig.add_trace(camera_pos_trace)

    # Add the camera data trace to the figure
    camera_data_trace = go.Scatter3d(
        x=[0], y=[0], z=[0], mode="lines", line=dict(color="blue", width=2)
    )
    fig.add_trace(camera_data_trace)

    # Set the plot limits and viewpoint
    fig.update_layout(
        scene_camera=dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=1.25, y=1.25, z=1.25),
        ),
        scene_aspectmode="cube",
    )


def update_plot():
    # Read the camera data
    X, Y, Z = read_camera_eo("./keyframes.pkl")

    # Update the camera position trace
    camera_pos_trace.x = [X[-1]]
    camera_pos_trace.y = [Y[-1]]
    camera_pos_trace.z = [Z[-1]]

    # Update the camera data trace
    camera_data_trace.x = X
    camera_data_trace.y = Y
    camera_data_trace.z = Z

    # Set the plot title
    fig.update_layout(title_text=f"Camera Trajectory (n={len(X)})")


### OPTIONS FOR EKF - Development temporarily interrupted, do not change values
T = 0.1
state_init = False

# Configuration file
CFG_FILE = "config.ini"

# Setup logging level
# TODO: use always logger instead of print
LOG_LEVEL = logging.INFO
utils.Inizialization.setup_logger(LOG_LEVEL)
logger = logging.getLogger("ColmapSLAM")

# Inizialize COLMAP SLAM problem
# NOTE: Initialization class moved to a new file in lib
init = utils.Inizialization(CFG_FILE)
cfg = init.inizialize()

# Initialize variables
keyframes_list = KeyFrameList()  # []  #
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

# Make plot
PLOT_TRJECTORY = False

# Setup keyframe selector
# TODO: move ALIKE config in config.ini file
alike_cfg = edict(
    {
        "model": "alike-s",
        "device": "cuda",
        "top_k": cfg.KFS_N_FEATURES,  # -1
        "scores_th": 0.2,
        "n_limit": 5000,
        "subpixel": False,
    }
)
keyframe_selector = KeyFrameSelector(
    keyframes_list=keyframes_list,
    last_keyframe_pointer=pointer,
    last_keyframe_delta=delta,
    keyframes_dir=cfg.KEYFRAMES_DIR,
    kfs_method="local_features",
    geometric_verification="pydegensac",  # "ransac",  #
    local_feature="ALIKE",  # "ORB",  #
    local_feature_cfg=alike_cfg,
    n_features=cfg.KFS_N_FEATURES,
    realtime_viz=False,  # True,
    viz_res_path=None,  # "kfs_matches",
)

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
if cfg.USE_SERVER == True:
    stream_proc = subprocess.Popen([cfg.LAUNCH_SERVER_PATH])
else:
    stream_proc = subprocess.Popen(["python3", "./simulator.py"])

# Set-up plotqq
# create_plot()
if PLOT_TRJECTORY:
    plot_proc = subprocess.Popen(["python3", "./plot.py"])

kfs_times = []

### MAIN LOOP
timer_global = utils.AverageTimer(logger=logger)
while True:
    # Get sorted image list available in imgs folders
    imgs = sorted(cfg.IMGS_FROM_SERVER.glob(f"*.{cfg.IMG_FORMAT}"))

    # If using the simulator, check if process is still alive, otherwise quit
    if cfg.USE_SERVER == False:
        if stream_proc.poll() is not None:
            logging.info("Simulator completed.")
            if PLOT_TRJECTORY:
                plot_proc.kill()
            break
    else:
        # Make exit condition when using server
        pass

    img_batch = []

    newer_imgs = False  # To control that new keyframes are added
    processed = 0  # Number of processed images

    # Keyframe selection
    if len(imgs) < 1:
        continue
    if len(imgs) < 2:
        # Set first frame as keyframe
        img0 = imgs[pointer]
        existing_keyframe_number = 0
        shutil.copy(
            img0,
            cfg.KEYFRAMES_DIR / f"{utils.Id2name(existing_keyframe_number)}",
        )
        camera_id = 1
        keyframes_list.add_keyframe(
            KeyFrame(
                img0,
                existing_keyframe_number,
                utils.Id2name(existing_keyframe_number),
                camera_id,
                pointer + delta + 1,
            )
        )
        continue
    elif len(imgs) >= 2:
        for c, img in enumerate(imgs):
            # Decide if new images are valid to be added to the sequential matching
            # Only new images found in the target folder are processed.
            # No more than MAX_IMG_BATCH_SIZE imgs are processed.
            if img in processed_imgs or c < 1 or processed >= cfg.MAX_IMG_BATCH_SIZE:
                continue

            print("")
            logger.info(f"pointer {pointer} c {c}")

            img1 = imgs[pointer]
            img2 = img

            old_n_keyframes = len(os.listdir(cfg.KEYFRAMES_DIR))

            (
                keyframes_list,
                pointer,
                delta,
                kfs_time,
            ) = keyframe_selector.run(img1, img2)
            # kfs_times.append(kfs_time)

            # Set if new keyframes are added
            new_n_keyframes = len(os.listdir(cfg.KEYFRAMES_DIR))
            if new_n_keyframes - old_n_keyframes > 0:
                newer_imgs = True
                img_batch.append(img)
                keyframe_obj = keyframes_list.get_keyframe_by_image_name(img)

                # Load exif data and store GNSS position if present
                # or load camera cooridnates from other sensors
                exif_data = []
                try:
                    exif_data = piexif.load(str(img2))
                except:
                    logger.error(
                        "Error loading exif data. Image file could be corrupted."
                    )

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

    # try:
    #     logging.info(
    #         f"Avearge KFS time: {np.array(kfs_times).mean():.4f} (std {np.array(kfs_times).std():.4f})"
    #     )
    # except:
    #     pass

    # INCREMENTAL RECONSTRUCTION
    kfrms = os.listdir(cfg.KEYFRAMES_DIR)
    kfrms.sort()

    if len(kfrms) >= cfg.MIN_KEYFRAME_FOR_INITIALIZATION and newer_imgs == True:
        timer = utils.AverageTimer(logger=logger)

        print()
        logger.info(f"DYNAMIC MATCHING WINDOW: {cfg.SEQUENTIAL_OVERLAP}")
        print()

        # FIRST LOOP IN COLMAP - INITIALIZATION
        if ended_first_colmap_loop == False:
            if cfg.CUSTOM_FEATURES == False:
                # Initialize an empty database
                if not os.path.exists(cfg.DATABASE):
                    subprocess.run(
                        [
                            str(cfg.COLMAP_EXE_PATH),
                            "database_creator",
                            "--database_path",
                            cfg.DATABASE,
                        ],
                        stdout=subprocess.DEVNULL,
                    )
                timer.update("DATABASE INITIALIZATION")

                # Feature extraction
                p = subprocess.run(
                    [
                        str(cfg.COLMAP_EXE_PATH),
                        "feature_extractor",
                        "--project_path",
                        cfg.CURRENT_DIR / "lib" / "sift_first_loop.ini",
                    ],
                    stdout=subprocess.DEVNULL,
                )
                timer.update("FEATURE EXTRACTION")

            elif cfg.CUSTOM_FEATURES == True:
                # Initialize an empty database
                if not os.path.exists(cfg.DATABASE):
                    subprocess.run(
                        [
                            str(cfg.COLMAP_EXE_PATH),
                            "database_creator",
                            "--database_path",
                            cfg.DATABASE,
                        ],
                        stdout=subprocess.DEVNULL,
                    )
                timer.update("DATABASE INITIALIZATION")

                # Feature extraction
                ExtractCustomFeatures.ExtractCustomFeatures(
                    cfg.CUSTOM_DETECTOR,
                    cfg.PATH_TO_LOCAL_FEATURES,
                    cfg.DATABASE,
                    cfg.KEYFRAMES_DIR,
                    keyframes_list,
                )
                timer.update("FEATURE EXTRACTION")

            # Sequential matcher
            subprocess.run(
                [
                    str(cfg.COLMAP_EXE_PATH),
                    "sequential_matcher",
                    "--project_path",
                    cfg.CURRENT_DIR / "lib" / "matcher.ini",
                ],
                stdout=subprocess.DEVNULL,
            )
            timer.update("SEQUENTIAL MATCHER")

            # Triangulation and BA
            subprocess.run(
                [
                    str(cfg.COLMAP_EXE_PATH),
                    "mapper",
                    "--project_path",
                    cfg.CURRENT_DIR / "lib" / "mapper_first_loop.ini",
                ],
                stdout=subprocess.DEVNULL,
            )
            timer.update("MAPPER")

            # Convert model from binary to txt
            subprocess.run(
                [
                    str(cfg.COLMAP_EXE_PATH),
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
            timer.update("MODEL CONVERSION")

            ended_first_colmap_loop = True

        # ALL COLMAP LOOPS AFTER THE FIRST - MODEL GROWTH
        elif ended_first_colmap_loop == True:
            if cfg.CUSTOM_FEATURES == False:
                # Feature extraction
                subprocess.call(
                    [
                        str(cfg.COLMAP_EXE_PATH),
                        "feature_extractor",
                        "--project_path",
                        cfg.CURRENT_DIR / "lib" / "sift.ini",
                    ],
                    stdout=subprocess.DEVNULL,
                )
                timer.update("FEATURE EXTRACTION")

            elif cfg.CUSTOM_FEATURES == True:
                # Feature extraction
                ExtractCustomFeatures.ExtractCustomFeatures(
                    cfg.CUSTOM_DETECTOR,
                    cfg.PATH_TO_LOCAL_FEATURES,
                    cfg.DATABASE,
                    cfg.KEYFRAMES_DIR,
                    keyframes_list,
                )
                timer.update("FEATURE EXTRACTION")

            # Sequential matcher
            # p = subprocess.Popen([COLMAP_EXE_PATH / "colmap", "sequential_matcher", "--project_path", CURRENT_DIR / "lib" / "matcher.ini"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
            if cfg.LOOP_CLOSURE_DETECTION == False:
                # Matching without loop closures (for all kind of local features)
                p = subprocess.call(
                    [
                        str(cfg.COLMAP_EXE_PATH),
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
                        str(cfg.COLMAP_EXE_PATH),
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
            timer.update("SEQUENTIAL MATCHER")

            # Triangulation and BA
            st_time = time.time()
            subprocess.call(
                [
                    str(cfg.COLMAP_EXE_PATH),
                    "mapper",
                    "--project_path",
                    cfg.CURRENT_DIR / "lib" / "mapper.ini",
                ],
                stdout=subprocess.DEVNULL,
            )
            timer.update("MAPPER")

            # Convert model from binary to txt
            p = subprocess.call(
                [
                    str(cfg.COLMAP_EXE_PATH),
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
            timer.update("MODEL CONVERSION")

        timer.print("COLMAP")

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
            keyframe_obj = keyframes_list.get_keyframe_by_image_name(image)
            if keyframe_obj.keyframe_id in list(oriented_dict.keys()):
                oriented_imgs_batch.append(image)
                keyframe_obj.set_oriented()

        # Define new reference img (pointer)
        print("list(oriented_dict.keys())")
        print(list(oriented_dict.keys()))
        last_oriented_keyframe = np.max(list(oriented_dict.keys()))
        print("last_oriented_keyframe", last_oriented_keyframe)
        keyframe_obj = keyframes_list.get_keyframe_by_id(last_oriented_keyframe)
        n_keyframes = len(os.listdir(cfg.KEYFRAMES_DIR))
        last_keyframe = keyframes_list.get_keyframe_by_id(n_keyframes - 1)
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
            keyframe_obj = keyframes_list.get_keyframe_by_id(ref_img_id)
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
                keyframe_obj = keyframes_list.get_keyframe_by_image_name(img_name)
                img_id = keyframe_obj.keyframe_id
                if img_id in oriented_dict_list:
                    list1.append(oriented_dict[img_id][1])
                    list2.append(
                        (keyframe_obj.slamX, keyframe_obj.slamY, keyframe_obj.slamZ)
                    )
            R_, t_, scale_factor_ = utils.Helmert(list1, list2, cfg.OS, cfg.DEBUG)

        # Apply rotantion matrix to move the updated photogrammetric model to the first model reference system
        for keyframe_id in oriented_dict_list:
            keyframe_obj = keyframes_list.get_keyframe_by_id(keyframe_id)
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
        #        # georeference slam with utils.Helmert transformation
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
        #        R, t, scale_factor = utils.Helmert(slam_coord, gnss_coord, OS, DEBUG)
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
        #            img_name = inverted_img_dict[utils.Id2name(img_id)]
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
        #                X1 = position_dict[inverted_img_dict[utils.Id2name(img_id - 2)]][
        #                    "slamX"
        #                ]
        #                Y1 = position_dict[inverted_img_dict[utils.Id2name(img_id - 2)]][
        #                    "slamY"
        #                ]
        #                Z1 = position_dict[inverted_img_dict[utils.Id2name(img_id - 2)]][
        #                    "slamZ"
        #                ]
        #                X2 = position_dict[inverted_img_dict[utils.Id2name(img_id - 1)]][
        #                    "slamX"
        #                ]
        #                Y2 = position_dict[inverted_img_dict[utils.Id2name(img_id - 1)]][
        #                    "slamY"
        #                ]
        #                Z2 = position_dict[inverted_img_dict[utils.Id2name(img_id - 1)]][
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

        # update_plot()

        img_batch = []
        oriented_imgs_batch = []

    timer_global.update(f"{len(kfrms)}")
    time.sleep(cfg.SLEEP_TIME)

average_loop_time = timer_global.get_average_time()
total_time = timer_global.get_total_time()
# timer_global.print("Timer global")

logging.info(f"Average loop time: {average_loop_time:.4f} s")
logging.info(f"Total time: {total_time:.4f} s")

print("Done.")
