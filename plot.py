# conda activate pillow
# https://stackoverflow.com/questions/4098131/how-to-update-a-plot-in-matplotlib

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import interactive
import numpy as np
import os
import time
import pickle

SLEEP = 0.5
condition = True



while condition == True:
    if os.path.exists("./keyframes.pkl"):
        plt.ion()
        interactive(True)
        fig = plt.figure()
        #fig, ax = plt.subplots(3, 1, subplot_kw={'projection' : '3d'}, constrained_layout=True, figsize=(8, 8))

        while condition == True:
            X = []
            Y = []
            Z = []
            if os.path.exists("./keyframes.pkl"):
                with open("./keyframes.pkl", 'rb') as f:
                    my_list = pickle.load(f)
                    for obj in my_list:
                        if obj.slamX != '-':
                            X.append(float(obj.slamX))
                        if obj.slamY != '-':
                            Y.append(float(obj.slamY))
                        if obj.slamZ != '-':
                            Z.append(float(obj.slamZ))

            
            ax = plt.axes(projection ='3d')
            #mngr = plt.get_current_fig_manager()
            #mngr.window.setGeometry(50,450,640, 495)

            MIN = min([min(X),min(Y),min(Z)])
            MAX = max([max(X),max(Y),max(Z)])

            #ax[0].cla()
            #ax[0].scatter(X, Y, Z, 'green')
            #ax[0].set_title('a')
            #ax[0].set_xticks([])
            #ax[0].set_yticks([])
            #ax[0].set_zticks([])
            #ax[0].view_init(azim=0, elev=90)
            #
            #ax[1].cla()
            #ax[1].scatter(X, Y, Z, 'blue')
            #ax[1].set_title('b')
            #ax[1].set_xticks([])
            #ax[1].set_yticks([])
            #ax[1].set_zticks([])
            #ax[1].view_init(azim=90, elev=0)
            
            ax.cla()
            ax.scatter(X, Y, Z, 'black')
            ax.set_title('c')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])           #ax[2].set_zticks(np.arange(MIN, MAX, (MAX-MIN)/10))
            ax.view_init(azim=0, elev=90)

            plt.show(block=False)
            plt.pause(SLEEP)
            #plt.clf()
            


            Total_imgs = len(os.listdir("./colmap_imgs"))
            N_oriented_cameras = len(X)
            print("Total: {}  Oriented: {}".format(Total_imgs, N_oriented_cameras))

    else:
        time.sleep(SLEEP)

#while condition == True:
#    if os.path.exists("./outs/loc.txt"):
#        plt.ion()
#        interactive(True)
#        fig = plt.figure()
#        #fig, ax = plt.subplots(3, 1, subplot_kw={'projection' : '3d'}, constrained_layout=True, figsize=(8, 8))
#
#        while condition == True:
#            X = []
#            Y = []
#            Z = []
#            if os.path.exists("./outs/loc.txt"):
#                with open("./outs/loc.txt", 'r') as file:
#                    lines = file.readlines()
#                    for line in lines[1:]:
#                        id, name, x, y, z, _ = line.split(" ", 5)
#                        X.append(float(x))
#                        Y.append(float(y))
#                        Z.append(float(z))
#
#            
#            ax = plt.axes(projection ='3d')
#            #mngr = plt.get_current_fig_manager()
#            #mngr.window.setGeometry(50,450,640, 495)
#
#            MIN = min([min(X),min(Y),min(Z)])
#            MAX = max([max(X),max(Y),max(Z)])
#
#            #ax[0].cla()
#            #ax[0].scatter(X, Y, Z, 'green')
#            #ax[0].set_title('a')
#            #ax[0].set_xticks([])
#            #ax[0].set_yticks([])
#            #ax[0].set_zticks([])
#            #ax[0].view_init(azim=0, elev=90)
#            #
#            #ax[1].cla()
#            #ax[1].scatter(X, Y, Z, 'blue')
#            #ax[1].set_title('b')
#            #ax[1].set_xticks([])
#            #ax[1].set_yticks([])
#            #ax[1].set_zticks([])
#            #ax[1].view_init(azim=90, elev=0)
#            
#            ax.cla()
#            ax.scatter(X, Y, Z, 'black')
#            ax.set_title('c')
#            ax.set_xticks([])
#            ax.set_yticks([])
#            ax.set_zticks([])           #ax[2].set_zticks(np.arange(MIN, MAX, (MAX-MIN)/10))
#            ax.view_init(azim=0, elev=90)
#
#            plt.show(block=False)
#            plt.pause(SLEEP)
#            #plt.clf()
#            
#
#
#            Total_imgs = len(os.listdir("./colmap_imgs"))
#            N_oriented_cameras = len(X)
#            print("Total: {}  Oriented: {}".format(Total_imgs, N_oriented_cameras))
#
#    else:
#        time.sleep(SLEEP)


