import subprocess
import os
import numpy as np
import time
import logging

from functools import wraps
from collections import OrderedDict


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f"Function {func.__name__} took {total_time:.4f} seconds")
        return result

    return timeit_wrapper


class AverageTimer:
    """Class to help manage printing simple timing of code execution."""

    def __init__(self, smoothing=0.3, logger=None):
        self.smoothing = smoothing
        self.times = OrderedDict()
        self.will_print = OrderedDict()
        self.logger = logger
        self.reset()

    def reset(self):
        now = time.time()
        self.start = now
        self.last_time = now
        for name in self.will_print:
            self.will_print[name] = False

    def update(self, name=""):
        now = time.time()
        dt = now - self.last_time
        if name in self.times:
            dt = self.smoothing * dt + (1 - self.smoothing) * self.times[name]
        self.times[name] = dt
        self.will_print[name] = True
        self.last_time = now

    def print(self, text="Timer"):
        total = 0.0
        msg = f"[{text}] | "
        for key in self.times:
            val = self.times[key]
            if self.will_print[key]:
                msg = msg + f"%s=%.3f, " % (key, val)
                total += val
        if self.logger is not None:
            self.logger.info(msg)
        else:
            logging.info(msg)

        self.reset()


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


def Helmert(aligned_coord, reference_coord, OS, DEBUG):
    """
    Helmert transformation using precompiled c++ library from CloudCompare
    """
    with open("./gt.txt", "w") as gt_file, open("./sl.txt", "w") as sl_file:
        count = 0
        for gt, sl in zip(reference_coord, aligned_coord):
            gt_file.write("{},{},{},{}\n".format(count, gt[0], gt[1], gt[2]))
            sl_file.write("{},{},{},{}\n".format(count, sl[0], sl[1], sl[2]))
            count += 1

    with open("./helemert.txt", "w") as output_file:
        if OS == "linux":
            subprocess.run(
                ["./AlignCC_for_linux/align", "./sl.txt", "./gt.txt"],
                stdout=output_file,
            )
        elif OS == "windows":
            subprocess.run(
                ["./AlignCC_for_windows/Align.exe", "./sl.txt", "./gt.txt"],
                stdout=output_file,
            )

    elms = []
    with open("./helemert.txt", "r") as helmert_file:
        lines = helmert_file.readlines()
        _, scale_factor = lines[1].strip().split(" ", 1)
        for line in lines[3:]:
            e1, e2, e3, e4 = line.strip().split(" ", 3)
            elms.extend((e1, e2, e3, e4))
    transf_matrix = np.array(elms, dtype="float32").reshape((4, 4))

    R = transf_matrix[:3, :3]
    t = transf_matrix[:3, 3].reshape((3, 1))
    os.remove("./gt.txt")
    os.remove("./sl.txt")
    os.remove("./helemert.txt")

    return R, t, float(scale_factor)
