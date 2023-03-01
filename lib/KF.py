# Da usare per update da GNSS e per quando è noto scale factor da SLAM
# Lo smooth ce l'abbiamo se consideriamo q più accurata dei sensori
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import interactive
import numpy as np
import os
import time

SLAM_SOLUTION = r"/home/luca/Scrivania/3DOM/Github_lcmrl/COLMAP_SLAM/outs/loc.txt"
T = 0.25
#lambd = 0.3

q = 0.0009
r = 0.1

#q = 0.0009
#r = 0.01

#q = 0.01
#r = 0.0009

F = np.array([
    [1, 0, 0, T, 0, 0],
    [0, 1, 0, 0, T, 0],
    [0, 0, 1, 0, 0 ,T],
    [0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1]])

H = np.array([
    [1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0]])

Q = np.array([
    [q, 0, 0, 0, 0, 0],
    [0, q, 0, 0, 0, 0],
    [0, 0, q, 0, 0, 0],
    [0, 0, 0, q, 0, 0],
    [0, 0, 0, 0, q, 0],
    [0, 0, 0, 0, 0, q]])

R = np.array([
    [r, 0, 0],
    [0, r, 0],
    [0, 0, r]])

P = np.array([
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0]])


def KalmanFilter(state_old, P, F, Q, obser, R, H):
    # Prediction phase
    status_predict = np.dot(F, state_old)
    print(status_predict)
    covariance_predict = np.dot(F, np.dot(P, F.T)) + Q

    # Update
    innovation = obser - np.dot(H, status_predict)
    #print(obser, status_predict)
    innov_cov = np.dot(H, np.dot(P, H.T)) + R
    gain = np.dot(covariance_predict, np.dot(H.T, np.linalg.inv(innov_cov)))

    #gain = np.ones((gain.shape[0], gain.shape[1]))

    state_new = status_predict + np.dot(gain, innovation)
    KH = np.dot(gain, H)
    assert np.shape(KH)[0] == np.shape(KH)[1]
    P_new = np.dot((np.identity(np.shape(KH)[0])-KH), covariance_predict)
    return state_new, P_new


if __name__ == "__main__":
    print("EKF")

    plt.ion()
    interactive(True)
    fig = plt.figure()

    Xslam = []
    Yslam = []
    Zslam = []

    Xkal = []
    Ykal = []
    Zkal = []

    img_dict = {}
    imgs = []

    with open(SLAM_SOLUTION, 'r') as slam_pos_txt:
        lines = slam_pos_txt.readlines()
        for line in lines[1:]:
            IMAGE_ID, X, Y, Z, _ = line.split(" ", 4)
            img_dict[int(IMAGE_ID[:-4])] = (X, Y, Z)
            imgs.append(int(IMAGE_ID[:-4]))
    
    imgs.sort()
    print(imgs)
    for i,img in enumerate(imgs[1:]):
        X, Y, Z = img_dict[img]
        print("img", img)
        print(X, Y, Z)
            
        if i == 0:
            X_old = float(X)#/lambd
            Y_old = float(Y)#/lambd
            Z_old = float(Z)#/lambd
            print("old", X_old, Y_old, Z_old)

        elif i == 1:
            X = float(X)#/lambd
            Y = float(Y)#/lambd
            Z = float(Z)#/lambd
            VX = (X-X_old)/(T)
            VY = (Y-Y_old)/(T)
            VZ = (Z-Z_old)/(T)
            
            state_old = np.array([[X, Y, Z, VX, VY, VZ]]).T
            print("state_old initialization: ", state_old)
        
        else:
            X = float(X)#/lambd
            Y = float(Y)#/lambd
            Z = float(Z)#/lambd
            Xslam.append(X)
            Yslam.append(Y)
            Zslam.append(Z)
            obser = np.array([[X, Y, Z]]).T
            state_new, P_new = KalmanFilter(state_old, P, F, Q, obser, R, H)
            print("obser", obser)
            print("state_old", state_old)
            print("state_new", state_new)
            
            state_old = state_new
            P = P_new
            Xkal.append(state_new[0,0])
            Ykal.append(state_new[1,0])
            Zkal.append(state_new[2,0])
            print(Xslam)
            print(Xkal)

            # plot
            ax = plt.axes(projection ='3d')
            MIN = min([min(Xkal),min(Ykal),min(Zkal)])
            MAX = max([max(Xkal),max(Ykal),max(Zkal)])
            ax.cla()
            ax.scatter(Xkal, Ykal, Zkal, c='r', label='kalman')
            ax.scatter(Xslam, Yslam, Zslam, c='b', label='slam')
            ax.set_title('c')
            #ax.set_xticks([])
            #ax.set_yticks([])
            #ax.set_zticks([])           
            ax.view_init(azim=0, elev=90)
            plt.legend(loc='upper left')
            plt.show(block=False)
            plt.pause(1)


                
