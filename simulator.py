import os
import shutil
import time
import configparser

from PIL import Image

STEP = 10

# from lib.utils import Id2name

# INPUT_DIR = r"data/MH_01_easy/mav0/cam0/data"
# OUTPUT_DIR = r"./imgs"

config = configparser.ConfigParser()
config.read("config.ini", encoding="utf-8")
INPUT_DIR = config["DEFAULT"]["SIMULATOR_IMG_DIR"]
OUTPUT_DIR = config["DEFAULT"]["IMGS_FROM_SERVER"]

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
for i in range(0, 1000000, STEP):
    img = imgs[i]
    # shutil.copy("{}/{}".format(INPUT_DIR, img), "{}/output{}.jpg".format(OUTPUT_DIR, i))
    im = Image.open("{}/{}".format(INPUT_DIR, img))
    rgb_im = im.convert("RGB")
    rgb_im.save("{}/{}.jpg".format(OUTPUT_DIR, img[:-4]))  # jpg
    time.sleep(1)
