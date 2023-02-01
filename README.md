# COLMAP_SLAM
SLAM based on COLMAP API for both windows and Linux OS.

If you want run the SLAM script inside docker:

```> xhost +local:*```

```> sudo docker run --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute -e DISPLAY=$DISPLAY -e XAUTHORITY=$XAUTHORITY --env="LIBGL_ALWAYS_INDIRECT=1" --env="QT_X11_NO_MITSHM=1" -v /tmp/.X11-unix:/tmp/.X11-unix --privileged -it --rm --name cmap4 -v /home:/home colmap_opencv```

To run COLMAP SLAM in docker:

```> python3 colmap_loop_linux.py```

To visualize the results in another shell:

```> python3 plot.py```