#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Name: Nwafor Solomon Chibuzo
Nuptun ID: UITK8W
Task 2: Filter moving objects.

Goal
1) remove ground using local height grid
2) keep only points in vehicle height band above ground
3) cluster height-band points in 2D
4) reject wall-like clusters using simple geometry rules
   - keep partial cars
   - reject long thin facade strips

Colors
gray  = all points
green = above ground
red   = kept as vehicle
"""

import argparse
from pathlib import Path
import numpy as np
import open3d as o3d

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "nuscenes"


def voxel_downsample_np(points_xyz: np.ndarray, voxel_size: float) -> np.ndarray:
    if points_xyz.shape[0] == 0:
        return points_xyz

    v = np.floor(points_xyz / voxel_size).astype(np.int64)
    order = np.lexsort((v[:, 2], v[:, 1], v[:, 0]))
    v = v[order]
    p = points_xyz[order]

    change = np.ones(len(v), dtype=bool)
    change[1:] = np.any(v[1:] != v[:-1], axis=1)
    idx = np.flatnonzero(change)
    idx_end = np.r_[idx[1:], len(v)]

    out = np.empty((len(idx), 3), dtype=np.float32)
    for k, (a, b) in enumerate(zip(idx, idx_end)):
        out[k] = p[a:b].mean(axis=0)

    return out


def get_aggregated_cloud_per_frame_voxel(nusc, scene_name, max_frames, per_frame_voxel):
    scene = next((s for s in nusc.scene if s["name"] == scene_name), None)
    if scene is None:
        raise ValueError(f"scene {scene_name} not found")

    points_list = []
    sample_token = scene["first_sample_token"]
    frame_idx = 0

    print(f"[{scene_name}] aggregating frames: {max_frames}")
    print(f"per-frame voxel: {per_frame_voxel} m")

    while sample_token:
        sample = nusc.get("sample", sample_token)
        lidar_token = sample["data"]["LIDAR_TOP"]
        sd = nusc.get("sample_data", lidar_token)

        calib = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
        ego = nusc.get("ego_pose", sd["ego_pose_token"])

        pc = LidarPointCloud.from_file(nusc.get_sample_data_path(lidar_token))

        sensor_to_ego = transform_matrix(calib["translation"], Quaternion(calib["rotation"]), inverse=False)
        ego_to_global = transform_matrix(ego["translation"], Quaternion(ego["rotation"]), inverse=False)
        pc.transform(ego_to_global @ sensor_to_ego)

        xyz = pc.points[:3, :].T.astype(np.float32)
        xyz = voxel_downsample_np(xyz, per_frame_voxel)
        points_list.append(xyz)

        frame_idx += 1
        if frame_idx >= max_frames:
            break
        sample_token = sample["next"]

    points = np.vstack(points_list) if len(points_list) else np.zeros((0, 3), dtype=np.float32)
    print(f"aggregated points: {points.shape[0]}")
    print(f"frames used: {frame_idx}")
    return points


def local_ground_heights(points: np.ndarray, cell_size: float, z_percentile: float) -> np.ndarray:
    xy = points[:, :2]
    cells = np.floor(xy / cell_size).astype(np.int64)
    uc, inv = np.unique(cells, axis=0, return_inverse=True)

    z = points[:, 2].astype(np.float32)
    z_ground = np.empty((uc.shape[0],), dtype=np.float32)

    for i in range(uc.shape[0]):
        zi = z[inv == i]
        if zi.size == 0:
            z_ground[i] = np.inf
        else:
            z_ground[i] = np.percentile(zi, z_percentile)

    return z_ground[inv]


def cluster_candidates_2d(points_xyz: np.ndarray, mask: np.ndarray, eps: float, min_points: int):
    idx = np.where(mask)[0]
    if idx.size == 0:
        return np.zeros((0,), dtype=np.int32), idx

    pts2 = points_xyz[idx].copy()
    pts2[:, 2] = 0.0

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(pts2)

    labels = np.array(
        pcd2.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True),
        dtype=np.int32
    )
    return labels, idx


def keep_vehicle_like_reject_walls(
    points: np.ndarray,
    labels: np.ndarray,
    idx: np.ndarray,
    h_min: float,
    h_max: float,
    wall_min_len: float,
    wall_max_thickness: float,
    wall_min_len_over_thickness: float,
    veh_max_len: float,
    veh_max_w: float,
    veh_max_h: float,
    veh_min_points: int,
):
    """
    Keep clusters that can be vehicles (including partial returns).
    Reject clusters that look like facade strips.

    Uses only geometry.
    No temporal logic.
    """
    kept = np.zeros((points.shape[0],), dtype=bool)
    if labels.size == 0 or labels.max() < 0:
        return kept

    for lab in range(labels.max() + 1):
        li = np.where(labels == lab)[0]
        if li.size == 0:
            continue

        if li.size < int(veh_min_points):
            continue

        gidx = idx[li]
        cluster = points[gidx]

        mins = cluster.min(axis=0)
        maxs = cluster.max(axis=0)
        dims = maxs - mins

        dx, dy, dz = float(dims[0]), float(dims[1]), float(dims[2])
        length_xy = max(dx, dy)
        width_xy = min(dx, dy)

        # cluster must lie inside height band
        if dz < float(h_min) or dz > float(min(h_max, veh_max_h)):
            continue

        # hard reject: enormous clusters
        if length_xy > float(veh_max_len):
            continue
        if width_xy > float(veh_max_w):
            continue

        # wall strip reject: long and thin in xy
        if length_xy >= float(wall_min_len) and width_xy <= float(wall_max_thickness):
            ratio = length_xy / max(width_xy, 1e-6)
            if ratio >= float(wall_min_len_over_thickness):
                continue

        # another facade reject: extremely long low-height strips
        if length_xy > 12.0 and dz < 1.5:
            continue

        kept[gidx] = True

    return kept


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="scene-0103")
    parser.add_argument("--max-frames", type=int, default=40)
    parser.add_argument("--per-frame-voxel", type=float, default=0.20)

    # ground model
    parser.add_argument("--ground-cell", type=float, default=0.4)
    parser.add_argument("--ground-percentile", type=float, default=12.0)

    # height band above ground (vehicle band)
    parser.add_argument("--h-min", type=float, default=0.25)
    parser.add_argument("--h-max", type=float, default=3.2)

    # clustering (on vehicle-band points)
    parser.add_argument("--db-eps", type=float, default=0.65)
    parser.add_argument("--db-min-points", type=int, default=8)

    # wall rejection knobs
    parser.add_argument("--wall-min-len", type=float, default=6.0)
    parser.add_argument("--wall-max-thickness", type=float, default=0.35)
    parser.add_argument("--wall-min-len-over-thickness", type=float, default=18.0)

    # vehicle-ish bounds (loose)
    parser.add_argument("--veh-max-len", type=float, default=15.0)
    parser.add_argument("--veh-max-w", type=float, default=5.0)
    parser.add_argument("--veh-max-h", type=float, default=3.2)
    parser.add_argument("--veh-min-points", type=int, default=25)

    # io
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument("--no-vis", action="store_true")

    args = parser.parse_args()

    nusc = NuScenes(version="v1.0-mini", dataroot=str(DATA_ROOT), verbose=False)

    pts = get_aggregated_cloud_per_frame_voxel(
        nusc=nusc,
        scene_name=args.scene,
        max_frames=args.max_frames,
        per_frame_voxel=args.per_frame_voxel,
    )
    if pts.shape[0] == 0:
        raise RuntimeError("no points loaded")

    print("step 1: local ground height")
    zg = local_ground_heights(pts, cell_size=args.ground_cell, z_percentile=args.ground_percentile)
    h = pts[:, 2].astype(np.float32) - zg

    above_ground = h > float(args.h_min)
    vehicle_band = (h > float(args.h_min)) & (h < float(args.h_max))

    print(f"above ground: {int(above_ground.sum())} / {pts.shape[0]}")
    print(f"vehicle band: {int(vehicle_band.sum())} / {pts.shape[0]}")

    print("step 2: cluster vehicle band in 2D")
    labels, idx = cluster_candidates_2d(pts, vehicle_band, eps=args.db_eps, min_points=args.db_min_points)

    print("step 3: keep vehicle, reject walls")
    kept_vehicle_like = keep_vehicle_like_reject_walls(
        points=pts,
        labels=labels,
        idx=idx,
        h_min=args.h_min,
        h_max=args.h_max,
        wall_min_len=args.wall_min_len,
        wall_max_thickness=args.wall_max_thickness,
        wall_min_len_over_thickness=args.wall_min_len_over_thickness,
        veh_max_len=args.veh_max_len,
        veh_max_w=args.veh_max_w,
        veh_max_h=args.veh_max_h,
        veh_min_points=args.veh_min_points,
    )

    print(f"kept vehicle points: {int(kept_vehicle_like.sum())} / {pts.shape[0]}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # outputs
    noground_pcd = o3d.geometry.PointCloud()
    noground_pcd.points = o3d.utility.Vector3dVector(pts[np.where(above_ground)[0]])
    o3d.io.write_point_cloud(str(out_dir / f"task2_{args.scene}_ground_filtered.pcd"), noground_pcd)

    band_pcd = o3d.geometry.PointCloud()
    band_pcd.points = o3d.utility.Vector3dVector(pts[np.where(vehicle_band)[0]])
    o3d.io.write_point_cloud(str(out_dir / f"task2_{args.scene}_vehicle_band.pcd"), band_pcd)

    kept_pcd = o3d.geometry.PointCloud()
    kept_pcd.points = o3d.utility.Vector3dVector(pts[np.where(kept_vehicle_like)[0]])
    o3d.io.write_point_cloud(str(out_dir / f"task2_{args.scene}_vehicle.pcd"), kept_pcd)

    print(f"saved: {out_dir / (f'task2_{args.scene}_ground_filtered.pcd')}")
    print(f"saved: {out_dir / (f'task2_{args.scene}_vehicle_band.pcd')}")
    print(f"saved: {out_dir / (f'task2_{args.scene}_vehicle.pcd')}")

    if args.no_vis:
        return

    # visualization
    # gray  = all
    # green = above ground
    # red   = kept vehicle
    pcd_all = o3d.geometry.PointCloud()
    pcd_all.points = o3d.utility.Vector3dVector(pts)

    colors = np.full((pts.shape[0], 3), 0.70, dtype=np.float32)
    colors[above_ground] = np.array([0.0, 0.8, 0.0], dtype=np.float32)
    colors[kept_vehicle_like] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    pcd_all.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries(
        [pcd_all],
        window_name="task2-step: gray=all, green=above ground, red=vehicle",
    )


if __name__ == "__main__":
    main()