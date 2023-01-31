# conda activate pillow
# python plot.py # run after running colmap loop
# https://stackoverflow.com/questions/4098131/how-to-update-a-plot-in-matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import interactive
import numpy as np
import os

plt.ion()
interactive(True)
fig = plt.figure()
condition = True

while condition == True:
    X = []
    Y = []
    Z = []
    if os.path.exists("./outs/loc.txt"):
        with open("./outs/loc.txt", 'r') as file:
            lines = file.readlines()
            for line in lines[1:]:
                id, x, y, z, _ = line.split(" ", 4)
                X.append(float(x))
                Y.append(float(y))
                Z.append(float(z))

    #for c in range(0, len(X)):
    #    print(X[c], Y[c], Z[c])
    
    ax = plt.axes(projection ='3d')

    mngr = plt.get_current_fig_manager()
    mngr.window.setGeometry(50,500,640, 545)

    ax.scatter(X, Y, Z, 'green')
    ax.set_title('3D line plot geeks for geeks')
    ax.set_xticks(np.arange(min([min(X),min(Y),min(Z)]), max([max(X),max(Y),max(Z)]), 1))
    ax.set_yticks(np.arange(min([min(X),min(Y),min(Z)]), max([max(X),max(Y),max(Z)]), 1))
    ax.set_zticks(np.arange(min([min(X),min(Y),min(Z)]), max([max(X),max(Y),max(Z)]), 1))
    
    plt.show(block=False)
    #plt.draw()
    plt.pause(3)
    plt.clf()
    #plt.close("all")
