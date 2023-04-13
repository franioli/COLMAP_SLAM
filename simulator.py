import os
import shutil
import time
import configparser
import sys
import logging
from pathlib import Path
from PIL import Image

# from lib.utils import Id2name

STEP = 1
SLEEP = 0.1
DEBUG = False


def run_simulator(imgs, output_dir="./imgs", ext="jpg"):
    for i, img in enumerate(imgs):
        # Process only every STEP-th image
        if i % STEP != 0:
            continue

        if DEBUG:
            print(f"processing {img} ({i}/{len(imgs)})")

        im = Image.open(img)
        rgb_im = im.convert("RGB")
        rgb_im.save(Path(output_dir) / f"{img.stem}.{ext}")
        time.sleep(SLEEP)


config = configparser.ConfigParser()
config.read("config.ini", encoding="utf-8")
input_dir = config["DEFAULT"]["SIMULATOR_IMG_DIR"]
output_dir = config["DEFAULT"]["IMGS_FROM_SERVER"]

ext = config["DEFAULT"]["IMG_FORMAT"]

imgs = sorted(Path(input_dir).glob("*"))
run_simulator(imgs, output_dir, ext)

logging.warning("No more images available")
sys.exit(0)

# imgs = os.listdir(INPUT_DIR)
# imgs.sort()

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
# for i in range(0, 1000000, STEP):
#     img = imgs[i]
#     # shutil.copy("{}/{}".format(INPUT_DIR, img), "{}/output{}.jpg".format(OUTPUT_DIR, i))
#     im = Image.open("{}/{}".format(INPUT_DIR, img))
#     rgb_im = im.convert("RGB")
#     rgb_im.save("{}/{}.jpg".format(OUTPUT_DIR, img[:-4]))  # jpg
#     time.sleep(1)


if __name__ == "__main__":
    input_dir = r"data/MH_01_easy/mav0/cam0/data"
    output_dir = r"./imgs"
    ext = config["DEFAULT"]["IMG_FORMAT"]

    imgs = sorted(Path(input_dir).glob("*"))
    run_simulator(imgs, output_dir, ext)
