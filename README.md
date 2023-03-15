# COLMAP_SLAM

SLAM based on COLMAP API for both Windows and Linux OS. The repository is under construction, if interested in the project you are free to join. Please note the repository is an adaptation of COLMAP to work in real-time. For code and license please refer to: <https://github.com/colmap/colmap>.

## EuRoC

Download dataset from <https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets>
For Machine Hall 01 use onli cam0

### Run in docker

If you want run the SLAM script inside docker, download the docker image od COLMAP and install the following dependencies:

```bash
pip3 install --upgrade pip
pip3 install opencv-contrib-python-headless (for docker, no GUI)
pip3 install pyquaternion
pip3 install scipy
pip3 install matplotlib
apt-get install -y python3-tk # to plot charts in docker
pip3 install piexif
pip3 install pyproj
pip3 install pymap3d
```

To run:

`> xhost +local:*`

`> sudo docker run --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute -e DISPLAY=$DISPLAY -e XAUTHORITY=$XAUTHORITY --env="LIBGL_ALWAYS_INDIRECT=1" --env="QT_X11_NO_MITSHM=1" -v /tmp/.X11-unix:/tmp/.X11-unix --privileged -it --rm --name cmap4 -v /home:/home -p 8080:8080 colmap_opencv`

To run COLMAP SLAM in docker:

`> python3 main.py`

### Install in Conda Environment

To install a an Anaconda environment (Linux, Windows, MacOS)

```bash
conda create -n colmap_slam python=3.10
conda activate colmap_slam
python3.8 -m pip install --upgrade pip
pip install -r requirements.txt
```

Change conf.ini
`> python3 main.py`

### Notes

Calibration
RaspCam 616.85,336.501,229.257,0.00335319
