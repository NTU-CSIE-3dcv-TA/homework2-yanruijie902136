import argparse
import os
from itertools import product

import cv2
import numpy as np
import pandas as pd

from constants import camera_matrix, dist_coeffs, image_h, image_w


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--rvecs", type=str, default="rvecs.npy")
    parser.add_argument("--tvecs", type=str, default="tvecs.npy")
    parser.add_argument("--cube_vertices", type=str, default="cube_vertices.npy")
    parser.add_argument("--fps", type=int, default=10)
    return parser.parse_args()


def make_cube_points(cube_vertices, n=10):
    face_indices = [[0, 1, 2, 3], [0, 1, 4, 5], [2, 3, 6, 7], [4, 5, 6, 7], [1, 3, 5, 7], [0, 2, 4, 6]]
    face_colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255], [0, 255, 255]]

    points = []
    colors = []
    for face, color in zip(face_indices, face_colors):
        vertices = cube_vertices[face]
        x = (vertices[1] - vertices[0]) / (n - 1)
        y = (vertices[2] - vertices[0]) / (n - 1)
        for i, j in product(range(n), range(n)):
            points.append(vertices[0] + i * x + j * y)
            colors.append(color)

    return np.asarray(points, dtype=np.float32), np.asarray(colors, dtype=np.uint8)


def draw_cube_points(image, rvec, tvec, points, colors):
    rotation_matrix, _ = cv2.Rodrigues(rvec)

    center = -rotation_matrix.T @ tvec
    sorted_indices = np.argsort([-np.linalg.norm(p - center) for p in points])
    points = points[sorted_indices]
    colors = colors[sorted_indices]

    points, _ = cv2.projectPoints(points, rvec, tvec, camera_matrix, dist_coeffs)
    points = points.reshape(-1, 2)

    for p, c in zip(points, colors):
        x, y = int(p[0]), int(p[1])
        if 0 <= x < image_w and 0 <= y < image_h:
            cv2.circle(image, (x, y), radius=5, color=c.tolist(), thickness=-1)

    return image


def main():
    args = parse_args()

    images_df = pd.read_pickle(os.path.join(args.data_dir, "images.pkl"))
    rvecs = np.load(os.path.join(args.rvecs))
    tvecs = np.load(os.path.join(args.tvecs))

    valid_images_df = images_df[images_df["NAME"].str.startswith("valid_img")]
    valid_images_df["num"] = valid_images_df["NAME"].str.extract(r"valid_img(\d+).jpg").astype(int)
    valid_images_df = valid_images_df.sort_values("num").reset_index(drop=True).drop(columns="num")

    valid_images = [
        cv2.imread(os.path.join(args.data_dir, "frames", image_name))
        for image_name in valid_images_df["NAME"]
    ]
    rvecs = rvecs[(valid_images_df["IMAGE_ID"] - 1).values]
    tvecs = tvecs[(valid_images_df["IMAGE_ID"] - 1).values]

    cube_vertices = np.load(args.cube_vertices)
    points, colors = make_cube_points(cube_vertices)

    video_writer = cv2.VideoWriter("ar.mp4", cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (image_w, image_h))
    for image, rvec, tvec in zip(valid_images, rvecs, tvecs):
        image = draw_cube_points(image, rvec, tvec, points, colors)
        video_writer.write(image)
    video_writer.release()


if __name__ == "__main__":
    main()
