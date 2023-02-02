import os
import shutil
import time

INPUT_DIR = r"/home/luca/Scrivania/3DOM/Github_lcmrl/COLMAP_SLAM/NuovaCartella"
OUTPUT_DIR = r"/home/luca/Scrivania/3DOM/Github_lcmrl/Server_Connection/c++_send_images/imgs"

for i in range (len(os.listdir(INPUT_DIR))):
    shutil.copy("{}/output{}.jpg".format(INPUT_DIR, i), "{}/output{}.jpg".format(OUTPUT_DIR, i))
    time.sleep(0.5)