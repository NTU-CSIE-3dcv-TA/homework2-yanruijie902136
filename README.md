[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/lyfclldM)
# Homework2

## Environment Setup

This project has been tested with Python 3.11. To install the required
packages, please run

```
python3 -m pip install -r requirements.txt
```

For problem 1, COLMAP is also required. We use COLMAP 3.12.6.

For problem 2, please download the dataset from [here](https://drive.google.com/u/0/uc?export=download&confirm=qrVw&id=1GrCpYJFc8IZM_Uiisq6e8UxwVMFvr4AJ).

You can view the homework demo [here](https://www.youtube.com/watch?v=W0JVzqYFp2s).

## Problem 1

Suppose we've saved the poind cloud file as `sparse.ply` after using COLMAP GUI. Then to create a mesh we run

```
python3 src/p1/mesh.py --pcd sparse.ply
```

## Problem 2

To estimate and visualize the camera poses, please run

```
python3 src/p2/pnp.py --data_dir data/
```

Two files will be created: `rvecs.npy` and `tvecs.npy`. These are the relative
poses of the cameras.

Next, we need to decide where to put the cube in the AR video. This is done by
running

```
python3 src/p2/transform_cube.py
```

This will produce `cube_vertices.npy`, containing the 8 vertices of the cube
in world coordinate frame.

Finally, we can create the AR video by running

```
python3 src/p2/ar.py \
    --data_dir data/ \
    --rvecs rvecs.npy \
    --tvecs tvecs.npy \
    --cube_vertices cube_vertices.npy \
    --fps 10
```

The video will be saved as `ar.mp4`.
