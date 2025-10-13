import argparse

import open3d as o3d


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pcd", type=str, default="pcd.ply")
    return parser.parse_args()


def main():
    args = parse_args()
    pcd = o3d.io.read_point_cloud(args.pcd)

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha=0.05)
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh])


if __name__ == "__main__":
    main()
