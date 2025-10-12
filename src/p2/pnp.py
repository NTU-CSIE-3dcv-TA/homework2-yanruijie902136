import argparse
import os
import random

import cv2
import numpy as np
import open3d as o3d
import pandas as pd
from scipy.spatial.transform import Rotation as R
from tqdm.auto import tqdm

from constants import camera_matrix, dist_coeffs


seed = 1428


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/")
    return parser.parse_args()


def set_seed():
    np.random.seed(seed)
    random.seed(seed)


def average_desc(train_df, points3d_df):
    train_df = train_df[["POINT_ID", "XYZ", "RGB", "DESCRIPTORS"]]
    desc_df = train_df.groupby("POINT_ID")["DESCRIPTORS"].apply(np.vstack)
    desc_df = desc_df.apply(lambda x: list(np.mean(x, axis=0)))
    desc_df = desc_df.reset_index()
    desc_df = desc_df.join(points3d_df.set_index("POINT_ID"), on="POINT_ID")
    return desc_df


def pnp_with_ransac(kp_query, desc_query, kp_model, desc_model):
    object_points = []
    image_points = []

    # Descriptor Matching and ratio test.
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc_query, desc_model, k=2)
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            object_points.append(kp_model[m.trainIdx])
            image_points.append(kp_query[m.queryIdx])

    success, rvec, tvec, _ = cv2.solvePnPRansac(
        np.array(object_points), np.array(image_points), camera_matrix, dist_coeffs)
    if not success:
        raise RuntimeError("PnP with RANSAC cannot find a solution")
    return rvec.flatten(), tvec.flatten()


def rotation_error(rvec1, rvec2):
    r1 = R.from_rotvec(rvec1)
    r2 = R.from_rotvec(rvec2)
    return (r2 * r1.inv()).magnitude()


def translation_error(tvec1, tvec2):
    return np.linalg.norm(tvec1 - tvec2)


def visualize(rvecs, tvecs, points3d_df):
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # Add point cloud.
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.vstack(points3d_df["XYZ"]))
    pcd.colors = o3d.utility.Vector3dVector(np.vstack(points3d_df["RGB"]) / 255.0)
    vis.add_geometry(pcd)

    # Add camera poses.
    for rvec, tvec in zip(rvecs, tvecs):
        rotation_matrix, _ = cv2.Rodrigues(rvec)

        points = np.array([[0, 0, 0], [-1, -1, 1], [-1, 1, 1], [1, 1, 1], [1, -1, 1]]) * 0.1
        points = (rotation_matrix.T @ (points.T - tvec[:, None])).T

        pyramid = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points),
            lines=o3d.utility.Vector2iVector([
                [0, 1], [0, 2], [0, 3], [0, 4],
                [1, 2], [2, 3], [3, 4], [4, 1],
            ]),
        )
        pyramid.colors = o3d.utility.Vector3dVector(np.tile([0, 0, 1], (8, 1)))
        vis.add_geometry(pyramid)

    vis.run()


def main():
    args = parse_args()

    set_seed()

    # Load data.
    images_df = pd.read_pickle(os.path.join(args.data_dir, "images.pkl"))
    train_df = pd.read_pickle(os.path.join(args.data_dir, "train.pkl"))
    points3d_df = pd.read_pickle(os.path.join(args.data_dir, "points3D.pkl"))
    point_desc_df = pd.read_pickle(os.path.join(args.data_dir, "point_desc.pkl"))

    # Process model keypoints descriptors.
    desc_df = average_desc(train_df, points3d_df)
    kp_model = np.array(desc_df["XYZ"].to_list())
    desc_model = np.array(desc_df["DESCRIPTORS"].to_list()).astype(np.float32)

    rvecs = []
    tvecs = []
    rerrors = []
    terrors = []

    image_ids = images_df["IMAGE_ID"].to_list()
    for image_id in tqdm(image_ids):
        # Load query keypoints and descriptors.
        points = point_desc_df.loc[point_desc_df["IMAGE_ID"] == image_id]
        kp_query = np.array(points["XY"].to_list())
        desc_query = np.array(points["DESCRIPTORS"].to_list()).astype(np.float32)

        rvec, tvec = pnp_with_ransac(kp_query, desc_query, kp_model, desc_model)
        rvecs.append(rvec)
        tvecs.append(tvec)

        # Get camera pose ground-truth.
        gt = images_df.loc[images_df["IMAGE_ID"] == image_id]
        rvec_gt = R.from_quat(gt[["QX", "QY", "QZ", "QW"]].values[0]).as_rotvec()
        tvec_gt = gt[["TX", "TY", "TZ"]].values[0]

        # Calculate errors.
        rerrors.append(rotation_error(rvec, rvec_gt))
        terrors.append(translation_error(tvec, tvec_gt))

    # Save the estimated camera poses.
    np.save("rvecs.npy", np.asarray(rvecs))
    np.save("tvecs.npy", np.asarray(tvecs))

    # Calculate median errors.
    print("Median of relative rotation angle differences (in radian):", np.median(rerrors))
    print("Median of translation differences:", np.median(terrors))

    # Visualize the trajectory, camera poses and 3d point cloud.
    visualize(rvecs, tvecs, points3d_df)


if __name__ == "__main__":
    main()
