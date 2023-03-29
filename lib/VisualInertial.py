import numpy as np
from scipy.integrate import cumtrapz
from scipy.linalg import block_diag
import pandas as pd

print("VisualInertial")

# Import imu data
imu_data = '/home/luca/Desktop/ION2023/EuRoC_MAV/MH_01_easy/mav0/imu0/data.csv'
df = pd.read_csv(imu_data, delimiter=',', header=0)

timestamp = df.iloc[:, 0].values
timestamp = timestamp.reshape((-1,1))
gyro_data = df.iloc[:, 1:4].values
acc_data = df.iloc[:, 4:7].values

# Define the state variables
x = np.zeros((7, 1))
P = np.eye(7)

# Define the measurement model
H = np.zeros((6, 7))
H[:3, :4] = 2 * np.array([[-x[3], -x[2], x[1]],
                          [x[2], -x[3], -x[0]],
                          [-x[1], x[0], -x[3]],
                          [x[0], x[1], x[2]]])
H[3:, 4:] = np.eye(3)

# Define the process model
F = np.eye(7)
F[:4, :4] = np.array([[1, -0.5*dt*omega[0], -0.5*dt*omega[1], -0.5*dt*omega[2]],
                      [0.5*dt*omega[0], 1, 0.5*dt*omega[2], -0.5*dt*omega[1]],
                      [0.5*dt*omega[1], -0.5*dt*omega[2], 1, 0.5*dt*omega[0]],
                      [0.5*dt*omega[2], 0.5*dt*omega[1], -0.5*dt*omega[0], 1]])
Q = np.eye(6) * sigma_process**2

# Define the measurement noise covariance
R = np.eye(6) * sigma_measurement**2

# Implement the extended Kalman filter algorithm
for i in range(1, len(timestamp)):
    # Prediction step
    omega = gyro_data[i-1:i, :].T
    dt = (timestamp[i] - timestamp[i-1]) / 1e9
    F[:4, 4:] = -0.5 * np.array([[x[1], x[2], x[3]],
                                 [-x[0], -x[3], x[2]],
                                 [-x[3], x[0], -x[1]],
                                 [x[2], -x[1], -x[0]]]) * dt
    x = F @ x
    P = F @ P @ F.T + Q

    # Correction step
    z = np.concatenate((acc_data[i-1:i, :].T, gyro_data[i-1:i, :].T))
    y = z - H @ x
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)
    x = x + K @ y
    P = (np.eye(7) - K @ H) @ P

    # Normalize quaternion
    q_norm = np.sqrt(x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2)
    x[:4] /= q_norm

quit()


# Define the state and measurement functions for the EKF
def f(x, dt, u):
    # x: [pos, vel, scale_factor, acc_bias, gyro_bias]
    # u: [acc, gyro]

    pos = x[:3]
    vel = x[3:6]
    scale_factor = x[6]
    acc_bias = x[7:10]
    gyro_bias = x[10:]

    acc = (u[0] - acc_bias)/scale_factor
    gyro = (u[1] - gyro_bias)

    vel = vel + dt*(acc + np.cross(gyro, vel))
    pos = pos + dt*vel + 0.5*dt**2*(acc + np.cross(gyro, vel))

    return np.concatenate((pos, vel, [scale_factor], acc_bias, gyro_bias))

def h(x):
    # x: [pos, vel, scale_factor, acc_bias, gyro_bias]

    pos = x[:3]
    return pos

# Load the accelerometer and gyroscope data
acc_data = np.load('acc_data.npy')
gyro_data = np.load('gyro_data.npy')

# Load the reference trajectory data
ref_data = np.load('ref_data.npy')

# Load the covariance matrices for the accelerometer and gyroscope measurements
acc_cov = np.load('acc_cov.npy')
gyro_cov = np.load('gyro_cov.npy')

# Load the covariance matrix for the other sensor's trajectory
ref_cov = np.load('ref_cov.npy')

# Define the sampling rate and time step
fs = 100.0 # Hz
dt = 1.0/fs

# Define the time vector
t = np.arange(len(acc_data))*dt

# Define the initial state and covariance matrices for the EKF
x = np.zeros(13)
P = block_diag(np.diag([1e-2, 1e-2, 1e-2]*3), np.diag([1e-4]*3), np.diag([1e-2]), np.diag([1e-4]*3), np.diag([1e-6]*3))

# Define the measurement and process noise covariance matrices for the EKF
R = ref_cov
Q = block_diag(np.matmul(np.matmul(orient[:-1], acc_cov), orient[:-1].T), np.matmul(np.matmul(orient[:-1], gyro_cov), orient[:-1].T))

# Define arrays to store the filtered state and covariance at each time step
filtered_states = np.zeros((len(ref_data), 13))
filtered_covs = np.zeros((len(ref_data), 13, 13))

# Run the EKF for each time step
for i in range(len(ref_data)):
    # Compute the acceleration and angular velocity using the measured data
    acc_measured = acc_data[i] - x[7:10]
    gyro_measured = gyro_data[i] - x[10:]
    ang_vel = gyro_measured

    ## Propagate the state and covariance using the process model
    #x = f(x, dt, [acc_measured, gyro_measured])
    #F = np.eye(13)
    #F[:3,3:6] = dt*np.eye(3)
    #F[3:6,6] = -dt*np.cross(ang_vel, np.eye(3)).flatten()
    #F[3:6,7: