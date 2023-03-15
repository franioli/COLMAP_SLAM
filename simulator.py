import os
import shutil
import time
from PIL import Image

INPUT_DIR = r"data/MH_01_easy/mav0/cam0/data"
OUTPUT_DIR = r"./imgs"


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


def run_simulator(INPUT_DIR: str, OUTPUT_DIR: str) -> bool:
    imgs = os.listdir(INPUT_DIR)
    imgs.sort()

    # for i in range (len(os.listdir(INPUT_DIR))):
    #    shutil.copy("{}/output{}.jpg".format(INPUT_DIR, i), "{}/output{}.jpg".format(OUTPUT_DIR, i))
    #    time.sleep(0.5)

    ##for i in range(140, len(os.listdir(INPUT_DIR))):
    # for i in range(140, 300):
    #    img = Id2name(i)
    #    shutil.copy("{}/{}".format(INPUT_DIR, img), "{}/output{}.jpg".format(OUTPUT_DIR, i-1))
    #    time.sleep(0.25)

    # for i in range(0, 600):
    #    shutil.copy("{}/img{}.jpg".format(INPUT_DIR, i), "{}/output{}.jpg".format(OUTPUT_DIR, i))
    #    time.sleep(0.25)

    ### EuRoC Machine Hall
    for i in range(0, 1000000, 20):
        img = imgs[i]
        # shutil.copy("{}/{}".format(INPUT_DIR, img), "{}/output{}.jpg".format(OUTPUT_DIR, i))
        im = Image.open("{}/{}".format(INPUT_DIR, img))
        rgb_im = im.convert("RGB")
        rgb_im.save("{}/{}.jpg".format(OUTPUT_DIR, img[:-4]))
        time.sleep(1)

    return True
