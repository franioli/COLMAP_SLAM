# COLMAP_SLAM
SLAM based on COLMAP API for both Windows and Linux OS. The repository is under construction, if interested in the project you are free to join. Please note the repository is an adaptation of COLMAP to work in real-time. For code and license please refer to: https://github.com/colmap/colmap.

### Run in Windows
Installation in a conda environment:
```
pip install piexif
pip install pyproj
pip install pymap3d
```

Change conf.ini
```> python3 main.py```


### Run in docker
If you want run the SLAM script inside docker, download the docker image od COLMAP and install the following dependencies:

```
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

```> xhost +local:*```

```> sudo docker run --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute -e DISPLAY=$DISPLAY -e XAUTHORITY=$XAUTHORITY --env="LIBGL_ALWAYS_INDIRECT=1" --env="QT_X11_NO_MITSHM=1" -v /tmp/.X11-unix:/tmp/.X11-unix --privileged -it --rm --name cmap4 -v /home:/home -p 8080:8080 colmap_opencv```

To run COLMAP SLAM in docker:

```> python3 main.py```

To visualize the results in another shell:

```> python3 plot.py```


### TO DO:
PER VELOCIZZARE TENERE CONTO CHE NON E' NECESSARIO PROCESSARE TUTTI I FRAMES MA SOLO I KEYFRAMES!!!
Server-Client c'è un problema con l'ordine dei file trovati nella cartella, vanno ordinati secondo un criterio
migliorare plot 3d
VELOCIZZARE TUTTO PER PROCESSARE PIU' FRAMES AL SECONDO
API COLMAP sequential_matcher è stranamente lenta rispetto alla GUI
ADESSO IL SEQUENTIAL OVERLAP DINAMICO NON FUNZIONA PIU', VA PASSATO A matcher.ini
https://medium.com/pythonland/6-things-you-need-to-know-to-run-commands-from-python-4ed5bc4c58a

### Notes:
Calibration
RaspCam 616.85,336.501,229.257,0.00335319
subprocess TUTORIAL https://www.bogotobogo.com/python/python_subprocess_module.php