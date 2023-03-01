# pip3 install --upgrade pip
# pip3 install opencv-contrib-python-headless (for docker, no GUI)
# pip3 install pyquaternion
# pip3 install scipy
# pip3 install matplotlib
# apt-get install -y python3-tk # to plot charts in docker
# pip3 install piexif
# pip3 install pyproj
# pip3 install pymap3d

# run in docker "colmap_opencv"
# python3 colmap_loop_linux.py

# DA FARE:
# PER VELOCIZZARE TENERE CONTO CHE NON E' NECESSARIO PROCESSARE TUTTI I FRAMES MA SOLO I KEYFRAMES!!!
# Server-Client c'è un problema con l'ordine dei file trovati nella cartella, vanno ordinati secondo un criterio
# migliorare plot 3d
# VELOCIZZARE TUTTO PER PROCESSARE PIU' FRAMES AL SECONDO
# API COLMAP sequential_matcher è stranamente lenta rispetto alla GUI
# ADESSO IL SEQUENTIAL OVERLAP DINAMICO NON FUNZIONA PIU', VA PASSATO A matcher.ini
# https://medium.com/pythonland/6-things-you-need-to-know-to-run-commands-from-python-4ed5bc4c58a1

# Calibration
# RaspCam 616.85,336.501,229.257,0.00335319

# subprocess TUTORIAL https://www.bogotobogo.com/python/python_subprocess_module.php

import configparser
import subprocess
import time
import shutil
import os
import numpy as np
from pathlib import Path
import cv2
import piexif
from pyquaternion import quaternion
from scipy.spatial.transform import Rotation as R
from scipy import linalg
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import interactive

from lib import database
from lib import static_rejection
from lib import export_cameras
from lib import ConvertGnssRefSystm
from lib import EKF
from lib import covariance_mat
from lib import ExtractCustomFeatures


T = 0.1
state_init = False
N_imgs_to_process = 30 #30


### FUNCTIONS
def Id2name(id):
    if id < 10:
        img_name = "00000{}.jpg".format(id)
    elif id < 100:
        img_name = "0000{}.jpg".format(id)
    elif id < 1000:
        img_name = "000{}.jpg".format(id)
    elif id < 10000:
        img_name = "00{}.jpg".format(id)
    elif id < 100000:
        img_name = "0{}.jpg".format(id)
    elif id < 1000000:
        img_name = "{}.jpg".format(id)
    return img_name

def Helmert(slam_coord, gnss_coord):
    with open('./gt.txt', 'w') as gt_file, open('./sl.txt', 'w') as sl_file:
        count = 0
        for gt, sl in zip(gnss_coord, slam_coord):
            gt_file.write("{},{},{},{}\n".format(count, gt[0], gt[1], gt[2]))
            sl_file.write("{},{},{},{}\n".format(count, sl[0], sl[1], sl[2]))
            count += 1
    output_file = open('./helemert.txt', 'w')
    subprocess.run(["./AlignCC_for_linux/align", "./sl.txt", "./gt.txt"], stdout=output_file)
    output_file.close()
    
    elms = []
    with open('./helemert.txt', 'r') as helmert_file:
        lines = helmert_file.readlines()
        _, scale_factor = lines[1].strip().split(" ", 1)
        for line in lines[3:]:
            e1, e2, e3, e4 = line.strip().split(" ", 3)
            elms.extend((e1, e2, e3, e4))
    transf_matrix = np.array(elms, dtype = 'float32').reshape((4, 4))
    print()
    print()
    print("transf_matrix", transf_matrix)
    R = transf_matrix[:3, :3]
    t = transf_matrix[:3, 3].reshape((3, 1))
    print(R, t)
    return R, t, float(scale_factor)

### MAIN STARTS HERE
CURRENT_DIR = Path(os.getcwd())
TEMP_DIR = CURRENT_DIR / "temp"
KEYFRAMES_DIR = CURRENT_DIR / "colmap_imgs"
OUT_FOLDER = CURRENT_DIR / "outs"
DATABASE = CURRENT_DIR / "outs" / "db.db"

# Import conf files
config = configparser.ConfigParser()
config.read(CURRENT_DIR / 'config.ini')
USE_SERVER = config['DEFAULT'].getboolean('USE_SERVER')
LAUNCH_SERVER_PATH = Path(config['DEFAULT']['LAUNCH_SERVER_PATH'])
DEBUG = config['DEFAULT'].getboolean('DEBUG')
STATIC_IMG_REJECTION_METHOD = config['DEFAULT']["STATIC_IMG_REJECTION_METHOD"] # 'radiometric' or 'root_sift'
SLEEP_TIME = float(config['DEFAULT']["SLEEP_TIME"])
LOOP_CYCLES = int(config['DEFAULT']["LOOP_CYCLES"])
COLMAP_EXE_PATH = Path(config['DEFAULT']["COLMAP_EXE_PATH"])
IMGS_FROM_SERVER = Path(config['DEFAULT']["IMGS_FROM_SERVER"]) #Path(r"/home/luca/Scrivania/3DOM/Github_lcmrl/Server_Connection/c++_send_images/imgs")
MAX_N_FEATURES = int(config['DEFAULT']["MAX_N_FEATURES"])
INITIAL_SEQUENTIAL_OVERLAP = int(config['DEFAULT']["INITIAL_SEQUENTIAL_OVERLAP"])
SEQUENTIAL_OVERLAP = int(config['DEFAULT']["SEQUENTIAL_OVERLAP"])
ONLY_SLAM = config['DEFAULT'].getboolean('ONLY_SLAM')
CUSTOM_FEATURES = config['DEFAULT'].getboolean('CUSTOM_FEATURES')
PATH_TO_LOCAL_FEATURES = Path(config['DEFAULT']["PATH_TO_LOCAL_FEATURES"])
CUSTOM_DETECTOR = config['DEFAULT']["CUSTOM_DETECTOR"]

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
#processed = 0
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

# If ground truth available
gt_dict = {}
#with open("/home/luca/Scrivania/3DOM/Github_lcmrl/COLMAP_SLAM/GroundTruth/sparse/ssssssssssssss.txt") as gt_file:
#    lines = gt_file.readlines()
#    for line in lines[2:]:
#        id, x, y, z, _ = line.split(" ", 4)
#        gt_dict[id] = (x, y, z)


### MAIN LOOP
if USE_SERVER == True:
    if not os.path.exists(IMGS_FROM_SERVER):
        os.makedirs(IMGS_FROM_SERVER)
    else:
        shutil.rmtree(IMGS_FROM_SERVER)
        os.makedirs(IMGS_FROM_SERVER)   
    p = subprocess.Popen([LAUNCH_SERVER_PATH])
    p = subprocess.Popen(["python3", "/home/luca/Scrivania/3DOM/Github_lcmrl/COLMAP_SLAM/plot.py"])

for i in range (LOOP_CYCLES):
    start_loop = time.time()
    imgs = os.listdir(IMGS_FROM_SERVER)
    imgs = sorted(imgs, key=lambda x: int(x[6:-4]))
    newer_imgs = False
    processed = 0
    
    # Choose if keeping the pair
    if len(imgs) < 2:
        print("[{}] len(imgs) < 2".format(i))

    elif len(imgs) >= 2:
        for c, img in enumerate(imgs):
            # Decide if new images are valid to be added to the sequential matching
            if img not in processed_imgs and c >= 1 and c != pointer and c > pointer+delta and processed < N_imgs_to_process:
                
                img1 = imgs[pointer]
                img2 = imgs[c]
                start = time.time()
                ref_matches, newer_imgs, total_imgs, img_dict, img_batch, pointer = static_rejection.StaticRejection(STATIC_IMG_REJECTION_METHOD, img1, img2, IMGS_FROM_SERVER, CURRENT_DIR, KEYFRAMES_DIR, COLMAP_EXE_PATH, MAX_N_FEATURES, ref_matches, DEBUG, newer_imgs, total_imgs, img_dict, img_batch, pointer) # pointer, delta, 
                end = time.time()
                print("STATIC CHECK {}s".format(end-start))
                processed_imgs.append(img)
                processed += 1
                

                # Load exif data and store GNSS position #################### Attenzione importare anche N E reference per sapere come è stata importata la latitudine e long
                exif_data = piexif.load("{}/imgs/{}".format(os.getcwd(), img2))

                if len(exif_data['GPS'].keys()) != 0:
                    print("img2", img2)
                    lat = exif_data['GPS'][2]
                    long = exif_data['GPS'][4]
                    alt = exif_data['GPS'][6]

                    enuX, enuY, enuZ = ConvertGnssRefSystm.CovertGnssRefSystm(lat, long, alt)

                    position_dict[img2] = {
                        'GPSLatitude' : exif_data['GPS'][2],
                        'GPSLongitude' : exif_data['GPS'][4],
                        'GPSAltitude' : exif_data['GPS'][6],
                        'enuX' : enuX,
                        'enuY' : enuY,
                        'enuZ' : enuZ,
                        'slamX' : '-',
                        'slamY' : '-',
                        'slamZ' : '-'
                        }   
                else:
                    if img2 in gt_dict.keys():
                        print("img2", img2)
                        enuX, enuY, enuZ = gt_dict[img2][0], gt_dict[img2][1], gt_dict[img2][2]

                        position_dict[img2] = {
                            'enuX' : enuX,
                            'enuY' : enuY,
                            'enuZ' : enuZ,
                            'slamX' : '-',
                            'slamY' : '-',
                            'slamZ' : '-'
                            }   
                    else:
                        position_dict[img2] = {
                            'enuX' : '-',
                            'enuY' : '-',
                            'enuZ' : '-',
                            'slamX' : '-',
                            'slamY' : '-',
                            'slamZ' : '-'
                            }   

    # LOCAL BUNDLE ADJUSTMENT
    kfrms = os.listdir(KEYFRAMES_DIR)
    kfrms.sort()
    
    if len(kfrms) >= 30 and newer_imgs == True: # 3 is mandatory or the pointer will not updated untill min of len(kfrms) is reached        
        if ended_first_colmap_loop == True:
            # subprocess examples:
            # subprocess.call([COLMAP_EXE_PATH / "colmap", "mapper", "--project_path", CURRENT_DIR / "lib" / "mapper.ini"])
            # subprocess.run([COLMAP_EXE_PATH / "colmap", "feature_extractor", "--project_path", CURRENT_DIR / "lib" / "sift_first_loop.ini"], stdout=subprocess.DEVNULL)
            # p = subprocess.Popen([COLMAP_EXE_PATH / "colmap", "sequential_matcher", "--database_path", DATABASE, "--SequentialMatching.overlap", "{}".format(SEQUENTIAL_OVERLAP), "--SequentialMatching.quadratic_overlap", "1"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
            # p.communicate()

            if CUSTOM_FEATURES == False:
                p = subprocess.call([COLMAP_EXE_PATH / "colmap", "feature_extractor", "--project_path", CURRENT_DIR / "lib" / "sift.ini"], stdout=subprocess.DEVNULL)
                #p.communicate()
            elif CUSTOM_FEATURES == True:
                ExtractCustomFeatures.ExtractCustomFeatures(CUSTOM_DETECTOR, PATH_TO_LOCAL_FEATURES, DATABASE, kfrms, img_dict, KEYFRAMES_DIR)
            #p = subprocess.Popen([COLMAP_EXE_PATH / "colmap", "sequential_matcher", "--project_path", CURRENT_DIR / "lib" / "matcher.ini"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
            p = subprocess.call([COLMAP_EXE_PATH / "colmap", "sequential_matcher", "--database_path", DATABASE, "--SequentialMatching.overlap", "{}".format(SEQUENTIAL_OVERLAP), "--SequentialMatching.quadratic_overlap", "1"], stdout=subprocess.DEVNULL)
            #p.communicate()
            st_time = time.time()
            subprocess.call([COLMAP_EXE_PATH / "colmap", "mapper", "--project_path", CURRENT_DIR / "lib" / "mapper.ini"])
            #p.communicate()
            end_time = time.time()
            p = subprocess.call([COLMAP_EXE_PATH / "colmap", "model_converter", "--input_path", OUT_FOLDER, "--output_path", OUT_FOLDER, "--output_type", "TXT"], stdout=subprocess.DEVNULL)
            #p.communicate()
            
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", end_time-st_time)

        elif ended_first_colmap_loop == False:
            if CUSTOM_FEATURES == False:
                if not os.path.exists(DATABASE): subprocess.run([COLMAP_EXE_PATH / "colmap", "database_creator", "--database_path", DATABASE], stdout=subprocess.DEVNULL)
                subprocess.run([COLMAP_EXE_PATH / "colmap", "feature_extractor", "--project_path", CURRENT_DIR / "lib" / "sift_first_loop.ini"], stdout=subprocess.DEVNULL)
            elif CUSTOM_FEATURES == True:
                if not os.path.exists(DATABASE): subprocess.run([COLMAP_EXE_PATH / "colmap", "database_creator", "--database_path", DATABASE], stdout=subprocess.DEVNULL)
                ExtractCustomFeatures.ExtractCustomFeatures(CUSTOM_DETECTOR, PATH_TO_LOCAL_FEATURES, DATABASE, kfrms, img_dict, KEYFRAMES_DIR)
            subprocess.run([COLMAP_EXE_PATH / "colmap", "sequential_matcher", "--project_path", CURRENT_DIR / "lib" / "matcher.ini"], stdout=subprocess.DEVNULL)
            subprocess.run([COLMAP_EXE_PATH / "colmap", "mapper", "--project_path", CURRENT_DIR / "lib" / "mapper_first_loop.ini"], stdout=subprocess.DEVNULL)
            subprocess.run([COLMAP_EXE_PATH / "colmap", "model_converter", "--input_path", OUT_FOLDER / "0", "--output_path", OUT_FOLDER, "--output_type", "TXT"], stdout=subprocess.DEVNULL)
            ended_first_colmap_loop = True

        lines, oriented_dict = export_cameras.ExportCameras(OUT_FOLDER / "images.txt", img_dict)
        with open(OUT_FOLDER / "loc.txt", 'w') as file:
            for line in lines:
                file.write(line)

        for im_input_format in img_batch:
            im_zero_format = img_dict[im_input_format]
            img_batch_n.append(int(im_zero_format[:-4]))
            if int(im_zero_format[:-4]) in list(oriented_dict.keys()):
                oriented_imgs_batch.append(int(im_zero_format[:-4]))

        #if len(oriented_imgs_batch) != 0:
        # Define new reference img (pointer)
        last_img_n = max(list(oriented_dict.keys())) #max(oriented_imgs_batch)
        max_img_n = max(img_batch_n)
        img_name = Id2name(last_img_n)
        inverted_img_dict = {v: k for k, v in img_dict.items()}
        for c, el in enumerate(imgs):
            #print(c,el)
            if el == inverted_img_dict[img_name]:
                pointer = c
        #pointer = imgs.index(inverted_img_dict[img_name])
        delta = max_img_n - last_img_n


        if delta != 0:
            SEQUENTIAL_OVERLAP = INITIAL_SEQUENTIAL_OVERLAP + 2*delta
        else:
            SEQUENTIAL_OVERLAP = INITIAL_SEQUENTIAL_OVERLAP

        end_loop = time.time()


        oriented_dict_list = list(oriented_dict.keys())
        oriented_dict_list.sort()
        out = open("./buttare1.txt", 'w')
        f1 = open("./f1.txt", 'w')
        f2 = open("./f2.txt", 'w')

        # Calculate transformation to report new slam solution on the old one
        if one_time == True:
            list1 = []
            list2 = []
            for img_id in oriented_dict_list:
                img_name = inverted_img_dict[Id2name(img_id)]
                if img_name in reference_imgs_dict.keys():
                    list1.append(oriented_dict[img_id][1])
                    f1.write("{},{},{},{}\n".format(img_id, oriented_dict[img_id][1][0], oriented_dict[img_id][1][1], oriented_dict[img_id][1][2]))
                    list2.append(reference_imgs_dict[img_name])
                    f2.write("{},{},{},{}\n".format(img_id, reference_imgs_dict[img_name][0], reference_imgs_dict[img_name][1], reference_imgs_dict[img_name][2]))
            R_, t_, scale_factor_ = Helmert(list1, list2)
            #R_, t_ = Helmert(list2, list1)
            print(R_, t_)
            #print("reference_imgs_dict", reference_imgs_dict)
            #print(img_dict); quit()

            ##################################################à
            #xxx = range(0, len(list1))
            #yyy1 = [el[2] for el in list1]
            #yyy2 = [el[2] for el in list2]
            #zzz = [0 for el in list1]
            #plt.ion()
            #interactive(True)
            #fig = plt.figure()
            #ax = plt.axes(projection ='3d')
            #MIN = min([min(Xslam),min(Yslam),min(Zslam)])
            #MAX = max([max(Xslam),max(Yslam),max(Zslam)])
            #ax.cla()
            #ax.scatter(xxx, yyy1, zzz, 'black')
            #ax.scatter(xxx, yyy2, zzz, 'red')
            #ax.set_title('c')
            ##ax.set_xticks([])
            ##ax.set_yticks([])
            ##ax.set_zticks([])           
            #ax.view_init(azim=0, elev=90)
            #plt.show(block=True)

        f1.close(); f2.close()


        for img_id in oriented_dict_list:    ################################################# OTTIMIZZARE, VANNO AGGIUNTE SOLO LE NUOVE IMMAGINI ORIENTATE, NO LOOP SU TUTTE LE IMMAGINI
            #print(img_id)
            #print(img_dict)
            #print(oriented_dict)
            #print(position_dict)
            img_name = inverted_img_dict[Id2name(img_id)]
            #print(img_name)
            #print(position_dict); quit()
            ref_img = list(position_dict.keys())[0]
            #print("ref_img", ref_img)

            if img_name in position_dict.keys():
                #print("img_name", img_name)
                if img_name == ref_img:
                    ref_img_id = img_id
                    quat1 = quaternion.Quaternion(oriented_dict[img_id][2][0])
                    t1 = oriented_dict[img_id][2][1]
                    position_dict[img_name]['slamX'] = 0.0
                    position_dict[img_name]['slamY'] = 0.0
                    position_dict[img_name]['slamZ'] = 0.0
                    out.write("{} 0.00 0.00 0.00 0.00 0.00 0.00 0.00 {} {}\n\n".format(img_id, 1, img_name))
                    #Xslam.append(0.0)
                    #Yslam.append(0.0)
                    #Zslam.append(0.0)
                
                elif img_name != ref_img and one_time == False:
                    #quat2 = quaternion.Quaternion(oriented_dict[img_id][2][0])
                    #t2 = oriented_dict[img_id][2][1]
                    #q2_1 = quat2 * quat1.inverse
                    #t2_1 = -np.dot((quat2 * quat1.inverse).rotation_matrix, t1) + t2
                    #q_matrix = q2_1.transformation_matrix
                    #q_matrix = q_matrix[0:3,0:3]
                    #camera_location = np.dot(-q_matrix.transpose(),t2_1)
                    vec_pos = np.array([[oriented_dict[img_id][1][0], oriented_dict[img_id][1][1], oriented_dict[img_id][1][2]]]).T
                    camera_location = vec_pos
                    position_dict[img_name]['slamX'] = camera_location[0,0]
                    position_dict[img_name]['slamY'] = camera_location[1,0]
                    position_dict[img_name]['slamZ'] = camera_location[2,0]
                    #Xslam.append(camera_location[0,0])
                    #Yslam.append(camera_location[1,0])
                    #Zslam.append(camera_location[2,0])
                    out.write("{},{},{}\n".format(camera_location[0,0], camera_location[1,0], camera_location[2,0]))
                    #out.write("{} {} {} {} {} {} {} {} {} {}\n\n".format(img_id, q2_1.scalar, q2_1.imaginary[0], q2_1.imaginary[1], q2_1.imaginary[2], t2_1[0][0], t2_1[1][0], t2_1[2][0], 1, img_name))
                    reference_imgs_dict[img_name] = (camera_location[0,0], camera_location[1,0], camera_location[2,0])
                
                elif img_name != ref_img and one_time == True:
                    #f1 = open("./f1.txt", 'w')
                    #f2 = open("./f2.txt", 'w')
                    #list1 = []
                    #list2 = []
                    #for img_id in oriented_dict_list:
                    #    print()
                    #    print()
                    #    print("img_id", img_id)
                    #    img_name = inverted_img_dict[Id2name(img_id)]
                    #    print("img_name", img_name)
                    #    if img_name in reference_imgs_dict.keys():
                    #        list1.append(oriented_dict[img_id][1])
                    #        f1.write("{},{},{}".format(img_id, oriented_dict[img_id][1][0], oriented_dict[img_id][1][1], oriented_dict[img_id][1][2]))
                    #        list2.append(reference_imgs_dict[img_name])
                    #        f2.write("{},{},{}".format(img_id, reference_imgs_dict[img_name][0], reference_imgs_dict[img_name][1], reference_imgs_dict[img_name][2]))
                    #R_, t_ = Helmert(list1, list2)
                    #f1.close()
                    #f2.close()

                    #print("one_time == True")
                    #quat2 = quaternion.Quaternion(oriented_dict[img_id][2][0])
                    #t2 = oriented_dict[img_id][2][1]
                    #q2_1 = quat2 * quat1.inverse
                    #t2_1 = -np.dot((quat2 * quat1.inverse).rotation_matrix, t1) + t2
                    #q_matrix = q2_1.transformation_matrix
                    #q_matrix = q_matrix[0:3,0:3]
                    #camera_location = np.dot(-q_matrix.transpose(),t2_1)
                    vec_pos = np.array([[oriented_dict[img_id][1][0], oriented_dict[img_id][1][1], oriented_dict[img_id][1][2]]]).T
                    #Tmx = np.array([
                    #    [R_[0,0], R_[0,1], R_[0,2], t_[0]],
                    #    [R_[1,0], R_[1,1], R_[1,2], t_[1]],
                    #    [R_[2,0], R_[2,1], R_[2,2], t_[2]],
                    #    [0, 0, 0, 1]
                    #])
                    #vec_omo = np.array([vec_pos[0,0], vec_pos[1,0], vec_pos[2,0], 1]).T
                    #aaaaaaaaa = np.matmul(Tmx, vec_omo)
                    #a = np.array([
                    #    [aaaaaaaaa[0]/aaaaaaaaa[3]],
                    #    [aaaaaaaaa[1]/aaaaaaaaa[3]],
                    #    [aaaaaaaaa[2]/aaaaaaaaa[3]]
                    #    ])
                    #R_ = Tmx[:3,:3]; t_ = Tmx[:3,3].reshape((3,1))
                    #print(Tmx)
                    #print(R_)
                    #print(t_)
                    
                    vec_pos_scaled = np.dot(R_, vec_pos) + t_
                    #t_ = t_.reshape((3,))
                    
                    #print("R_", R_); print("t_",t_); print(vec_pos)
                    #print()
                    #print()
                    #print("a", a)
                    #print("vec_pos_scaled", vec_pos_scaled)
                    position_dict[img_name]['slamX'] = vec_pos_scaled[0,0]
                    position_dict[img_name]['slamY'] = vec_pos_scaled[1,0]
                    position_dict[img_name]['slamZ'] = vec_pos_scaled[2,0]
                    #Xslam.append(vec_pos_scaled[0,0])
                    #Yslam.append(vec_pos_scaled[1,0])
                    #Zslam.append(vec_pos_scaled[2,0])
                    #out.write("{} {} {} {} {} {} {} {} {} {}\n\n".format(img_id, q2_1.scalar, q2_1.imaginary[0], q2_1.imaginary[1], q2_1.imaginary[2], t2_1[0][0], t2_1[1][0], t2_1[2][0], 1, img_name))
                    out.write("{},{},{}\n".format(vec_pos_scaled[0,0], vec_pos_scaled[1,0], vec_pos_scaled[2,0]))
        
        
        out.close()
        #if one_time == True: quit()
        one_time = True
        
        

        #plt.ion()
        #interactive(True)
        #fig = plt.figure()
        #ax = plt.axes(projection ='3d')
        #MIN = min([min(Xslam),min(Yslam),min(Zslam)])
        #MAX = max([max(Xslam),max(Yslam),max(Zslam)])
        #ax.cla()
        #ax.scatter(Xslam, Yslam, Zslam, 'black')
        #ax.set_title('c')
        ##ax.set_xticks([])
        ##ax.set_yticks([])
        ##ax.set_zticks([])           
        #ax.view_init(azim=0, elev=90)
        #plt.show(block=True)

        #print(position_dict)



        

        if ONLY_SLAM != True:

            # INITIALIZATION SCALE FACTOR AND KALMAN FILTER
            if len(kfrms) == 30:
                # For images with both slam and gnss solution
                # georeference slam with Helmert transformation
                slam_coord = []
                gnss_coord = []
                for img in position_dict:
                    if position_dict[img]['enuX'] != '-':
                        gnss_coord.append((position_dict[img]['enuX'], position_dict[img]['enuY'], position_dict[img]['enuZ']))
                        slam_coord.append((position_dict[img]['slamX'], position_dict[img]['slamY'], position_dict[img]['slamZ']))
                #print(slam_coord, gnss_coord)
                
                R, t, scale_factor = Helmert(slam_coord, gnss_coord)
                #print(R, t)
    
                #Store positions
                slam_coord = []
                for img in position_dict:
                    slam_coord.append((position_dict[img]['slamX'], position_dict[img]['slamY'], position_dict[img]['slamZ']))
                for pos in slam_coord:
                    if pos[0] != '-':
                        pos = np.array([[pos[0]], [pos[1]], [pos[2]]])
                        scaled_pos = np.dot(R, pos) + t
                        Xslam.append(scaled_pos[0,0])
                        Yslam.append(scaled_pos[1,0])
                        Zslam.append(scaled_pos[2,0])
    
                #plt.ion()
                #interactive(True)
                #fig = plt.figure()
                #ax = plt.axes(projection ='3d')
                #MIN = min([min(Xslam),min(Yslam),min(Zslam)])
                #MAX = max([max(Xslam),max(Yslam),max(Zslam)])
                #ax.cla()
                #ax.scatter(Xslam, Yslam, Zslam, 'black')
                #ax.set_title('c')
                ##ax.set_xticks([])
                ##ax.set_yticks([])
                ##ax.set_zticks([])           
                #ax.view_init(azim=0, elev=90)
                #plt.show(block=True)
                #quit()
    
            elif len(kfrms) > 30:
                oriented_imgs_batch.sort()
                for img_id in oriented_imgs_batch:
                    #print(img_id)
                    img_name = inverted_img_dict[Id2name(img_id)]
                    # Positions in Sdr of the reference img
                    x = position_dict[img_name]['slamX']
                    y = position_dict[img_name]['slamY']
                    z = position_dict[img_name]['slamZ']
                    observation = np.array([[x], [y], [z]])
                    scaled_observation = np.dot(R, observation) + t
                    Xslam.append(scaled_observation[0,0])
                    Yslam.append(scaled_observation[1,0])
                    Zslam.append(scaled_observation[2,0])
    
                    if state_init == False:
                        X1 = position_dict[inverted_img_dict[Id2name(img_id-2)]]['slamX']
                        Y1 = position_dict[inverted_img_dict[Id2name(img_id-2)]]['slamY']
                        Z1 = position_dict[inverted_img_dict[Id2name(img_id-2)]]['slamZ']
                        X2 = position_dict[inverted_img_dict[Id2name(img_id-1)]]['slamX']
                        Y2 = position_dict[inverted_img_dict[Id2name(img_id-1)]]['slamY']
                        Z2 = position_dict[inverted_img_dict[Id2name(img_id-1)]]['slamZ']
                        X_1 = np.array([[X1, Y1, Z1]]).T
                        X_2 = np.array([[X2, Y2, Z2]]).T
                        X_1 = np.dot(R, X_1) + t
                        X_2 = np.dot(R, X_2) + t
                        V = (X_2-X_1)/T
                        state_old = np.array([[X_2[0,0], X_2[1,0], X_2[2,0], V[0,0], V[1,0], V[2,0], 1]]).T
                        state_init = True
                        P = covariance_mat.Pini()
    
                    # Smooth with EKF
                    #state_new, P_new, lambd = EKF.ExtendedKalmanFilter(state_old, P, covariance_mat.F(T), covariance_mat.Q(0.0009, 0.0001), scaled_observation, covariance_mat.R(0.1))
                    state_new, P_new, lambd = EKF.ExtendedKalmanFilter(state_old, P, covariance_mat.F(T), covariance_mat.Q(0.0001, 0.000001), scaled_observation, covariance_mat.R(0.01))
                    
                    Xkf.append(state_old[0,0])
                    Ykf.append(state_old[1,0])
                    Zkf.append(state_old[2,0])
                    state_old = state_new
                    P = P_new
                    #print("lambd", lambd)
    
    
                    plt.ion()
                    interactive(True)
                    fig = plt.figure()
                    ax = plt.axes(projection ='3d')
                    MIN = min([min(Xslam),min(Yslam),min(Zslam)])
                    MAX = max([max(Xslam),max(Yslam),max(Zslam)])
                    ax.cla()
                    ax.scatter(Xslam, Yslam, Zslam, 'black')
                    ax.scatter(Xkf, Ykf, Zkf, 'red')
                    ax.set_title('c')
                    #ax.set_xticks([])
                    #ax.set_yticks([])
                    #ax.set_zticks([])           
                    ax.view_init(azim=0, elev=90)
                    plt.show(block=True)
                
    
    
                        # predict new position with EKF (to calibrate scale factor so more accuracy on Q and less on R)
                            # if GNSS present
                                # Use the known prediction from slam and apply KF
    
                        # Print scale factor
            
                



        img_batch = []
        oriented_imgs_batch = []

        print("LOOP TIME {}s\n\n\n".format(end_loop-start_loop))
        

    time.sleep(SLEEP_TIME)






    
