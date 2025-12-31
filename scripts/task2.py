#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Task 2 baby-step (improved, full script)

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
red   = kept as vehicle-like
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

    print("step 3: keep vehicle-like, reject walls")
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

    print(f"kept vehicle-like points: {int(kept_vehicle_like.sum())} / {pts.shape[0]}")

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
    # red   = kept vehicle-like
    pcd_all = o3d.geometry.PointCloud()
    pcd_all.points = o3d.utility.Vector3dVector(pts)

    colors = np.full((pts.shape[0], 3), 0.70, dtype=np.float32)
    colors[above_ground] = np.array([0.0, 0.8, 0.0], dtype=np.float32)
    colors[kept_vehicle_like] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    pcd_all.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries(
        [pcd_all],
        window_name="task2-step: gray=all, green=above ground, red=vehicle-like",
    )


if __name__ == "__main__":
    main()



# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# """
# Task 2 baby-step (improved)

# Goal
# 1) remove ground using local height grid
# 2) keep only points in vehicle height band above ground
# 3) remove wall-like structures using 2D clustering + shape filters
#    - reject clusters that look like vertical planes (tall vs horizontal extent)
#    - reject clusters that are too long (facade strips)
#    - keep clusters that match vehicle footprint and height

# Colors
# gray  = all points
# green = above ground
# red   = kept as vehicle-like
# """

# import argparse
# from pathlib import Path
# import numpy as np
# import open3d as o3d

# from nuscenes.nuscenes import NuScenes
# from nuscenes.utils.data_classes import LidarPointCloud
# from nuscenes.utils.geometry_utils import transform_matrix
# from pyquaternion import Quaternion


# REPO_ROOT = Path(__file__).resolve().parents[1]
# DATA_ROOT = REPO_ROOT / "nuscenes"


# def voxel_downsample_np(points_xyz: np.ndarray, voxel_size: float) -> np.ndarray:
#     if points_xyz.shape[0] == 0:
#         return points_xyz

#     v = np.floor(points_xyz / voxel_size).astype(np.int64)
#     order = np.lexsort((v[:, 2], v[:, 1], v[:, 0]))
#     v = v[order]
#     p = points_xyz[order]

#     change = np.ones(len(v), dtype=bool)
#     change[1:] = np.any(v[1:] != v[:-1], axis=1)
#     idx = np.flatnonzero(change)
#     idx_end = np.r_[idx[1:], len(v)]

#     out = np.empty((len(idx), 3), dtype=np.float32)
#     for k, (a, b) in enumerate(zip(idx, idx_end)):
#         out[k] = p[a:b].mean(axis=0)

#     return out


# def get_aggregated_cloud_per_frame_voxel(nusc, scene_name, max_frames, per_frame_voxel):
#     scene = next((s for s in nusc.scene if s["name"] == scene_name), None)
#     if scene is None:
#         raise ValueError(f"scene {scene_name} not found")

#     points_list = []
#     sample_token = scene["first_sample_token"]
#     frame_idx = 0

#     print(f"[{scene_name}] aggregating frames: {max_frames}")
#     print(f"per-frame voxel: {per_frame_voxel} m")

#     while sample_token:
#         sample = nusc.get("sample", sample_token)
#         lidar_token = sample["data"]["LIDAR_TOP"]
#         sd = nusc.get("sample_data", lidar_token)

#         calib = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
#         ego = nusc.get("ego_pose", sd["ego_pose_token"])

#         pc = LidarPointCloud.from_file(nusc.get_sample_data_path(lidar_token))

#         sensor_to_ego = transform_matrix(calib["translation"], Quaternion(calib["rotation"]), inverse=False)
#         ego_to_global = transform_matrix(ego["translation"], Quaternion(ego["rotation"]), inverse=False)
#         pc.transform(ego_to_global @ sensor_to_ego)

#         xyz = pc.points[:3, :].T.astype(np.float32)
#         xyz = voxel_downsample_np(xyz, per_frame_voxel)
#         points_list.append(xyz)

#         frame_idx += 1
#         if frame_idx >= max_frames:
#             break
#         sample_token = sample["next"]

#     points = np.vstack(points_list) if len(points_list) else np.zeros((0, 3), dtype=np.float32)
#     print(f"aggregated points: {points.shape[0]}")
#     print(f"frames used: {frame_idx}")
#     return points


# def local_ground_heights(points: np.ndarray, cell_size: float, z_percentile: float) -> np.ndarray:
#     xy = points[:, :2]
#     cells = np.floor(xy / cell_size).astype(np.int64)
#     uc, inv = np.unique(cells, axis=0, return_inverse=True)

#     z = points[:, 2].astype(np.float32)
#     z_ground = np.empty((uc.shape[0],), dtype=np.float32)

#     for i in range(uc.shape[0]):
#         zi = z[inv == i]
#         if zi.size == 0:
#             z_ground[i] = np.inf
#         else:
#             z_ground[i] = np.percentile(zi, z_percentile)

#     return z_ground[inv]


# def cluster_candidates_2d(points_xyz: np.ndarray, mask: np.ndarray, eps: float, min_points: int):
#     idx = np.where(mask)[0]
#     if idx.size == 0:
#         return np.zeros((0,), dtype=np.int32), idx

#     pts2 = points_xyz[idx].copy()
#     pts2[:, 2] = 0.0

#     pcd2 = o3d.geometry.PointCloud()
#     pcd2.points = o3d.utility.Vector3dVector(pts2)

#     labels = np.array(
#         pcd2.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True),
#         dtype=np.int32
#     )
#     return labels, idx


# def filter_wall_like_clusters(
#     points: np.ndarray,
#     labels: np.ndarray,
#     idx: np.ndarray,
#     h_min: float,
#     h_max: float,
#     # vehicle-ish footprint constraints (loose)
#     min_len: float,
#     max_len: float,
#     min_w: float,
#     max_w: float,
#     # wall rejection
#     max_cluster_len: float,
#     max_height_over_len: float,
# ):
#     kept = np.zeros((points.shape[0],), dtype=bool)
#     if labels.size == 0 or labels.max() < 0:
#         return kept

#     for lab in range(labels.max() + 1):
#         li = np.where(labels == lab)[0]
#         if li.size == 0:
#             continue

#         gidx = idx[li]
#         cluster = points[gidx]

#         mins = cluster.min(axis=0)
#         maxs = cluster.max(axis=0)
#         dims = maxs - mins

#         dx, dy, dz = float(dims[0]), float(dims[1]), float(dims[2])
#         length_xy = max(dx, dy)
#         width_xy = min(dx, dy)

#         # 1) hard height band on the cluster itself
#         if not (h_min <= dz <= h_max):
#             continue

#         # 2) reject facade strips: too long in xy
#         if length_xy > max_cluster_len:
#             continue

#         # 3) reject vertical-plane-ish: too tall relative to horizontal span
#         # walls often have dz comparable to or larger than their local span
#         if (dz / max(length_xy, 1e-6)) > max_height_over_len:
#             continue

#         # 4) keep vehicle-ish footprints (loose)
#         # cars can be partially observed, so keep it permissive
#         footprint_ok = (min_len <= length_xy <= max_len) and (min_w <= width_xy <= max_w)
#         if not footprint_ok:
#             continue

#         kept[gidx] = True

#     return kept


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--scene", default="scene-0103")
#     parser.add_argument("--max-frames", type=int, default=40)
#     parser.add_argument("--per-frame-voxel", type=float, default=0.20)

#     # ground model
#     parser.add_argument("--ground-cell", type=float, default=0.4)
#     parser.add_argument("--ground-percentile", type=float, default=12.0)

#     # height band above ground (vehicle band)
#     parser.add_argument("--h-min", type=float, default=0.25)
#     parser.add_argument("--h-max", type=float, default=3.2)

#     # clustering (on vehicle-band points)
#     parser.add_argument("--db-eps", type=float, default=0.55)
#     parser.add_argument("--db-min-points", type=int, default=12)

#     # footprint constraints (loose)
#     parser.add_argument("--min-len", type=float, default=1.2)
#     parser.add_argument("--max-len", type=float, default=14.0)
#     parser.add_argument("--min-w", type=float, default=0.6)
#     parser.add_argument("--max-w", type=float, default=4.2)

#     # wall rejection knobs
#     parser.add_argument("--max-cluster-len", type=float, default=15.0) #18.0
#     parser.add_argument("--max-height-over-len", type=float, default=0.85)

#     # io
#     parser.add_argument("--out-dir", default="output")
#     parser.add_argument("--no-vis", action="store_true")

#     args = parser.parse_args()

#     nusc = NuScenes(version="v1.0-mini", dataroot=str(DATA_ROOT), verbose=False)
#     pts = get_aggregated_cloud_per_frame_voxel(nusc, args.scene, args.max_frames, args.per_frame_voxel)

#     if pts.shape[0] == 0:
#         raise RuntimeError("no points loaded")

#     print("step 1: local ground height")
#     zg = local_ground_heights(pts, cell_size=args.ground_cell, z_percentile=args.ground_percentile)
#     h = pts[:, 2].astype(np.float32) - zg

#     above_ground = h > float(args.h_min)
#     vehicle_band = (h > float(args.h_min)) & (h < float(args.h_max))

#     print(f"above ground: {int(above_ground.sum())} / {pts.shape[0]}")
#     print(f"vehicle band: {int(vehicle_band.sum())} / {pts.shape[0]}")

#     print("step 2: cluster vehicle band in 2D")
#     labels, idx = cluster_candidates_2d(pts, vehicle_band, eps=args.db_eps, min_points=args.db_min_points)

#     print("step 3: reject wall-like clusters by shape")
#     kept_vehicle_like = filter_wall_like_clusters(
#         points=pts,
#         labels=labels,
#         idx=idx,
#         h_min=args.h_min,
#         h_max=args.h_max,
#         min_len=args.min_len,
#         max_len=args.max_len,
#         min_w=args.min_w,
#         max_w=args.max_w,
#         max_cluster_len=args.max_cluster_len,
#         max_height_over_len=args.max_height_over_len,
#     )

#     print(f"kept vehicle-like points: {int(kept_vehicle_like.sum())} / {pts.shape[0]}")

#     out_dir = Path(args.out_dir)
#     out_dir.mkdir(parents=True, exist_ok=True)

#     # outputs
#     noground_pcd = o3d.geometry.PointCloud()
#     noground_pcd.points = o3d.utility.Vector3dVector(pts[np.where(above_ground)[0]])
#     o3d.io.write_point_cloud(str(out_dir / f"{args.scene}_noground.pcd"), noground_pcd)

#     band_pcd = o3d.geometry.PointCloud()
#     band_pcd.points = o3d.utility.Vector3dVector(pts[np.where(vehicle_band)[0]])
#     o3d.io.write_point_cloud(str(out_dir / f"{args.scene}_vehicle_band.pcd"), band_pcd)

#     kept_pcd = o3d.geometry.PointCloud()
#     kept_pcd.points = o3d.utility.Vector3dVector(pts[np.where(kept_vehicle_like)[0]])
#     o3d.io.write_point_cloud(str(out_dir / f"{args.scene}_vehicle_like.pcd"), kept_pcd)

#     print(f"saved: {out_dir / (args.scene + '_noground.pcd')}")
#     print(f"saved: {out_dir / (args.scene + '_vehicle_band.pcd')}")
#     print(f"saved: {out_dir / (args.scene + '_vehicle_like.pcd')}")

#     if args.no_vis:
#         return

#     # visualization
#     # gray  = all
#     # green = above ground
#     # red   = kept vehicle-like
#     pcd_all = o3d.geometry.PointCloud()
#     pcd_all.points = o3d.utility.Vector3dVector(pts)

#     colors = np.full((pts.shape[0], 3), 0.70, dtype=np.float32)
#     colors[above_ground] = np.array([0.0, 0.8, 0.0], dtype=np.float32)
#     colors[kept_vehicle_like] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
#     pcd_all.colors = o3d.utility.Vector3dVector(colors)

#     o3d.visualization.draw_geometries(
#         [pcd_all],
#         window_name="task2 baby-step improved: gray=all, green=above ground, red=vehicle-like"
#     )


# if __name__ == "__main__":
#     main()



# #!/usr/bin/env python3
# """
# Task 2: Moving Object Filtering (Unsupervised)
# Approch: 
# 1. Temporal Voxelization: Identify "new" points vs "old" points.
# 2. RANSAC: Mathematically find and remove the road (Fixes lane-artifact issues).
# 3. DBSCAN: Cluster remaining points into objects.
# 4. Heuristics: Filter clusters based on Vehicle Dimensions (Length, Width, Height).
# """

# import argparse
# from pathlib import Path
# import numpy as np
# import open3d as o3d
# from nuscenes.nuscenes import NuScenes
# from nuscenes.utils.data_classes import LidarPointCloud
# from nuscenes.utils.geometry_utils import transform_matrix
# from pyquaternion import Quaternion

# # Update this path to your local nuScenes directory
# REPO_ROOT = Path(__file__).resolve().parents[1]
# DATA_ROOT = REPO_ROOT / "nuscenes"

# def get_aggregated_cloud(nusc, scene_name, max_frames):
#     scene = next((s for s in nusc.scene if s["name"] == scene_name), None)
#     if not scene: raise ValueError(f"Scene {scene_name} not found.")

#     points_list = []
#     frame_ids_list = []
#     sample_token = scene["first_sample_token"]
#     frame_idx = 0

#     print(f"[{scene_name}] Aggregating {max_frames if max_frames else 'all'} frames...")
#     while sample_token:
#         sample = nusc.get("sample", sample_token)
#         lidar_token = sample["data"]["LIDAR_TOP"]
#         sd = nusc.get("sample_data", lidar_token)
#         calib = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
#         ego = nusc.get("ego_pose", sd["ego_pose_token"])

#         pc = LidarPointCloud.from_file(nusc.get_sample_data_path(lidar_token))
        
#         # Sensor -> Ego -> Global
#         sensor_to_ego = transform_matrix(calib["translation"], Quaternion(calib["rotation"]), inverse=False)
#         ego_to_global = transform_matrix(ego["translation"], Quaternion(ego["rotation"]), inverse=False)
#         pc.transform(ego_to_global @ sensor_to_ego)

#         xyz = pc.points[:3, :].T
#         points_list.append(xyz)
#         frame_ids_list.append(np.full(xyz.shape[0], frame_idx, dtype=np.int32))

#         frame_idx += 1
#         if max_frames and frame_idx >= max_frames: break
#         sample_token = sample["next"]

#     return np.vstack(points_list), np.concatenate(frame_ids_list)

# def moving_object_pipeline(points, frame_ids):
#     # --- 0. Pre-processing: Downsample to protect RAM ---
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(points)
#     pcd = pcd.voxel_down_sample(voxel_size=0.15)
#     down_points = np.asarray(pcd.points)

#     # --- 1. Temporal Analysis ---
#     # We use a 0.5m voxel to allow for slight GNSS drift in static objects
#     print("Step 1: Analyzing temporal consistency...")
#     voxel_size_t = 0.5
#     v_coords = np.floor(down_points / voxel_size_t).astype(np.int64)
#     # This is a trick to calculate temporal occupancy efficiently
#     unique_v_f = np.unique(np.hstack([v_coords, frame_ids[:len(down_points), None]]), axis=0)
#     uv, counts = np.unique(unique_v_f[:, :3], axis=0, return_counts=True)
#     static_set = set(map(tuple, uv[counts >= 2]))
#     is_new = np.array([tuple(v) not in static_set for v in v_coords])

#     # --- 2. Ground Removal (RANSAC) ---
#     # This identifies the road plane and ignores it, solving the 'Lanes' problem
#     print("Step 2: Removing road surface via RANSAC...")
#     plane_model, inliers = pcd.segment_plane(distance_threshold=0.25, ransac_n=3, num_iterations=1000)
#     obj_indices = np.delete(np.arange(len(down_points)), inliers)
#     obj_points = down_points[obj_indices]

#     # --- 3. Clustering ---
#     print(f"Step 3: Clustering {len(obj_points)} potential objects...")
#     pcd_obj = o3d.geometry.PointCloud()
#     pcd_obj.points = o3d.utility.Vector3dVector(obj_points)
#     labels = np.array(pcd_obj.cluster_dbscan(eps=0.9, min_points=10, print_progress=True))
    
#     # --- 4. Car Hunter Heuristics ---
#     final_moving_mask = np.zeros(len(down_points), dtype=bool)
#     for i in range(labels.max() + 1):
#         l_idx = np.where(labels == i)[0]
#         g_idx = obj_indices[l_idx]
#         cluster = obj_points[l_idx]
        
#         dims = cluster.max(axis=0) - cluster.min(axis=0)
#         length = max(dims[0], dims[1])
#         width = min(dims[0], dims[1])

#         # VEHICLE SIGNATURE:
#         # Height: 0.5m - 3.2m (Includes SUVs/Vans)
#         # Length: 2.0m - 12.0m (Includes Buses)
#         # Width: < 4.0m
#         is_vehicle = (0.5 < dims[2] < 3.2) and (2.0 < length < 12.0) and (width < 4.0)
        
#         if is_vehicle:
#             # Check if at least 15% of the cluster is "temporally new"
#             if np.mean(is_new[g_idx]) > 0.15:
#                 final_moving_mask[g_idx] = True

#     print(f"Final Count: {np.sum(final_moving_mask)} moving points identified.")
#     return down_points, final_moving_mask

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--scene", default="scene-0103")
#     parser.add_argument("--max-frames", type=int, default=40)
#     args = parser.parse_args()

#     nusc = NuScenes(version="v1.0-mini", dataroot=str(DATA_ROOT), verbose=False)

#     # Task 1: Aggregate
#     points, frame_ids = get_aggregated_cloud(nusc, args.scene, args.max_frames)

#     # Task 2: Filter
#     processed_pts, moving_mask = moving_object_pipeline(points, frame_ids)

#     # Save & Visualize
#     out_dir = Path("output")
#     out_dir.mkdir(exist_ok=True)
    
#     final_pcd = o3d.geometry.PointCloud()
#     final_pcd.points = o3d.utility.Vector3dVector(processed_pts)
#     colors = np.full(processed_pts.shape, 0.5)
#     colors[moving_mask] = [1, 0, 0] # Red for moving
#     final_pcd.colors = o3d.utility.Vector3dVector(colors)

#     o3d.io.write_point_cloud(str(out_dir / f"{args.scene}_static_map.pcd"), 
#                              final_pcd.select_by_index(np.where(~moving_mask)[0]))
    
#     o3d.visualization.draw_geometries([final_pcd], window_name="Task 2 Final: Red=Moving")

# if __name__ == "__main__":
#     main()


# #!/usr/bin/env python3
# """
# Task 2 (Final Car Hunter): Physics + Shape Recognition.
# Strategy:
# 1. Downsample to save RAM.
# 2. Cluster objects.
# 3. IDENTIFY CARS based on Dimensions (Width/Length/Height).
# 4. Apply "Smart Voting":
#    - If it looks like a car, be lenient on movement check.
#    - If it doesn't look like a car, delete it.
# """

# import argparse
# from pathlib import Path
# import numpy as np
# import open3d as o3d
# from nuscenes.nuscenes import NuScenes
# from nuscenes.utils.data_classes import LidarPointCloud
# from nuscenes.utils.geometry_utils import transform_matrix
# from pyquaternion import Quaternion

# REPO_ROOT = Path(__file__).resolve().parents[1]
# DATA_ROOT = REPO_ROOT / "nuscenes"

# def get_aggregated_cloud(nusc, scene_name, max_frames):
#     scene = next((s for s in nusc.scene if s["name"] == scene_name), None)
#     if not scene: raise ValueError("Scene not found")
    
#     points_list = []
#     frame_ids_list = []
#     sample_token = scene["first_sample_token"]
#     frame_idx = 0
    
#     print(f"[{scene_name}] Aggregating frames...")
#     while sample_token:
#         sample = nusc.get("sample", sample_token)
#         lidar_token = sample["data"]["LIDAR_TOP"]
#         sd = nusc.get("sample_data", lidar_token)
#         calib = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
#         ego = nusc.get("ego_pose", sd["ego_pose_token"])
        
#         pc = LidarPointCloud.from_file(nusc.get_sample_data_path(lidar_token))
#         sensor_to_ego = transform_matrix(calib["translation"], Quaternion(calib["rotation"]), inverse=False)
#         ego_to_global = transform_matrix(ego["translation"], Quaternion(ego["rotation"]), inverse=False)
#         pc.transform(ego_to_global @ sensor_to_ego)
        
#         xyz = pc.points[:3, :].T
#         points_list.append(xyz)
#         frame_ids_list.append(np.full(xyz.shape[0], frame_idx, dtype=np.int32))
        
#         frame_idx += 1
#         if max_frames and frame_idx >= max_frames: break
#         sample_token = sample["next"]
        
#     return np.vstack(points_list), np.concatenate(frame_ids_list)

# def downsample_arrays(points, frame_ids, voxel_size=0.15):
#     print(f"Downsampling cloud (Leaf: {voxel_size}m)...")
#     voxels = np.floor(points / voxel_size).astype(np.int64)
#     _, unique_indices = np.unique(voxels, axis=0, return_index=True)
#     print(f"  -> Reduced to {len(unique_indices)} points.")
#     return points[unique_indices], frame_ids[unique_indices]

# def task2_pipeline(points, frame_ids):
#     # 1. Downsample (Safety)
#     points, frame_ids = downsample_arrays(points, frame_ids, voxel_size=0.15)

#     # 2. Temporal Check (Voxel 0.5m)
#     # This determines which points are "suspicious" (potentially moving)
#     print("Step 1: Temporal Check (0.5m)...")
#     voxel_size = 0.5 
#     voxels = np.floor(points / voxel_size).astype(np.int64)
#     data = np.hstack([voxels, frame_ids.reshape(-1, 1)])
#     unique_voxel_frames = np.unique(data, axis=0)
#     unique_voxels, counts = np.unique(unique_voxel_frames[:, :3], axis=0, return_counts=True)
    
#     static_coords = unique_voxels[counts >= 2]
#     static_set = set(map(tuple, static_coords))
#     is_temporally_new = np.array([tuple(v) not in static_set for v in voxels])
#     print(f"  -> {np.sum(is_temporally_new)} suspicious points.")

#     # 3. Ground Cut (Strict)
#     # Remove road (< 0.3m) so cars separate from the floor
#     non_ground_mask = points[:, 2] > 0.3
#     object_indices = np.where(non_ground_mask)[0]
#     object_points = points[object_indices]
    
#     print(f"Step 2: Clustering {len(object_points)} objects...")
    
#     # 4. Clustering (DBSCAN)
#     # eps=1.0 connects car parts together
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(object_points)
#     labels = np.array(pcd.cluster_dbscan(eps=1.0, min_points=5, print_progress=True))
#     max_label = labels.max()
#     print(f"  -> Found {max_label + 1} objects.")

#     # 5. THE CAR HUNTER LOOP
#     final_moving_mask = np.zeros(len(points), dtype=bool)
#     kept_count = 0
    
#     for i in range(max_label + 1):
#         global_indices = object_indices[labels == i]
#         cluster_pts = points[global_indices]
        
#         # Calculate Dimensions
#         min_box = cluster_pts.min(axis=0)
#         max_box = cluster_pts.max(axis=0)
#         dims = max_box - min_box # [dx, dy, dz]
        
#         # --- SHAPE CHECKS ---
#         # A. Width/Length Check (Is it car sized?)
#         # Standard Car: ~1.8m wide, ~4.5m long. 
#         # We allow a range: Width [0.5 - 3.0], Length [1.5 - 12.0] (includes buses)
#         is_car_width = (0.5 < dims[0] < 4.0) or (0.5 < dims[1] < 4.0)
#         is_car_length = (1.5 < dims[0] < 12.0) or (1.5 < dims[1] < 12.0)
        
#         # B. Height Check (Crucial)
#         # Cars are NOT taller than 2.2m.
#         is_car_height = (0.5 < dims[2] < 2.2)
        
#         # C. Wall Check
#         # If it's huge (> 15m), it's a wall.
#         is_huge = (dims[0] > 15.0) or (dims[1] > 15.0)
        
#         if is_huge: continue
#         if not is_car_height: continue # Too flat (ground) or too tall (tree)
#         if not (is_car_width and is_car_length): continue # Too tiny (pole) or weird shape

#         # --- MOTION CHECK (Smart Voting) ---
#         cluster_temporal_flags = is_temporally_new[global_indices]
#         moving_ratio = np.mean(cluster_temporal_flags)
        
#         # LOGIC: 
#         # If it fits the Car Shape perfectly, we trust even weak movement (10%).
#         # This brings back the "Patchy" or "Gray" cars.
#         if moving_ratio > 0.10:
#             final_moving_mask[global_indices] = True
#             kept_count += 1
            
#     print(f"  -> Identified {kept_count} CARS.")
#     return points, final_moving_mask

# def save_and_viz(points, is_moving, scene_name):
#     print("Saving results...")
#     out_dir = REPO_ROOT / "output"
#     out_dir.mkdir(parents=True, exist_ok=True)
    
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(points)
    
#     o3d.io.write_point_cloud(str(out_dir / f"{scene_name}_static.pcd"), pcd.select_by_index(np.where(~is_moving)[0]))
#     o3d.io.write_point_cloud(str(out_dir / f"{scene_name}_moving.pcd"), pcd.select_by_index(np.where(is_moving)[0]))
    
#     print("Visualizing...")
#     colors = np.full(points.shape, 0.5)
#     colors[is_moving] = [1, 0, 0] # Red
#     pcd.colors = o3d.utility.Vector3dVector(colors)
    
#     vis = o3d.visualization.Visualizer()
#     vis.create_window(window_name="Task 2: Car Hunter")
#     vis.add_geometry(pcd)
#     vis.get_render_option().background_color = np.asarray([0, 0, 0])
#     vis.run()
#     vis.destroy_window()

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--scene", default="scene-0103")
#     parser.add_argument("--max-frames", type=int, default=150)
#     args = parser.parse_args()
    
#     nusc = NuScenes(version="v1.0-mini", dataroot=str(DATA_ROOT), verbose=False)
#     points, frame_ids = get_aggregated_cloud(nusc, args.scene, args.max_frames)
    
#     points, is_moving = task2_pipeline(points, frame_ids)
#     save_and_viz(points, is_moving, args.scene)

# if __name__ == "__main__":
#     main()