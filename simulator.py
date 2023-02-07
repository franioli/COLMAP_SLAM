import os
import shutil
import time

INPUT_DIR = r"/home/luca/Scrivania/3DOM/Github_lcmrl/COLMAP_SLAM/RILIEVI/FBK2/colmap_imgs" #
OUTPUT_DIR = r"/home/luca/Scrivania/3DOM/Github_lcmrl/Server_Connection/c++_send_images/imgs"


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


imgs = os.listdir(INPUT_DIR)
imgs.sort()

#for i in range (len(os.listdir(INPUT_DIR))):
#    shutil.copy("{}/output{}.jpg".format(INPUT_DIR, i), "{}/output{}.jpg".format(OUTPUT_DIR, i))
#    time.sleep(0.5)

for i in range(1, len(os.listdir(INPUT_DIR))):
    img = Id2name(i)
    shutil.copy("{}/{}".format(INPUT_DIR, img), "{}/output{}.jpg".format(OUTPUT_DIR, i-1))
    time.sleep(0.5)