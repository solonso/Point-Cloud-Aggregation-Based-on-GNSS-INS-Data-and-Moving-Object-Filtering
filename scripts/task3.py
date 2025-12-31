#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Task 3: point cloud colorization (no annotations)

Pipeline
1) aggregate lidar to global
2) run task2 geometry filter to get filtered_mask
3) colorize points using camera images
4) export:
   - colored_static.pcd (filtered points removed)
   - colored_full_black.pcd (optional: filtered kept but pure black)

No manual annotation used.
"""

import argparse
from pathlib import Path
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

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


def get_aggregated_cloud_per_frame_voxel(
    nusc: NuScenes,
    scene_name: str,
    max_frames: int,
    per_frame_voxel: float,
):
    scene = next((s for s in nusc.scene if s["name"] == scene_name), None)
    if scene is None:
        raise ValueError(f"scene {scene_name} not found")

    points_list = []
    sample_tokens = []

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
        sample_tokens.append(sample_token)

        frame_idx += 1
        if frame_idx >= max_frames:
            break
        sample_token = sample["next"]

    points = np.vstack(points_list) if len(points_list) else np.zeros((0, 3), dtype=np.float32)

    print(f"aggregated points: {points.shape[0]}")
    print(f"frames used: {frame_idx}")
    return points, sample_tokens


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

        if dz < float(h_min) or dz > float(min(h_max, veh_max_h)):
            continue

        if length_xy > float(veh_max_len):
            continue
        if width_xy > float(veh_max_w):
            continue

        if length_xy >= float(wall_min_len) and width_xy <= float(wall_max_thickness):
            ratio = length_xy / max(width_xy, 1e-6)
            if ratio >= float(wall_min_len_over_thickness):
                continue

        if length_xy > 12.0 and dz < 1.5:
            continue

        kept[gidx] = True

    return kept


def task2_filter_mask(
    points: np.ndarray,
    ground_cell: float,
    ground_percentile: float,
    h_min: float,
    h_max: float,
    db_eps: float,
    db_min_points: int,
    wall_min_len: float,
    wall_max_thickness: float,
    wall_min_len_over_thickness: float,
    veh_max_len: float,
    veh_max_w: float,
    veh_max_h: float,
    veh_min_points: int,
):
    zg = local_ground_heights(points, cell_size=ground_cell, z_percentile=ground_percentile)
    h = points[:, 2].astype(np.float32) - zg

    vehicle_band = (h > float(h_min)) & (h < float(h_max))
    print(f"task2: vehicle band: {int(vehicle_band.sum())} / {points.shape[0]}")

    labels, idx = cluster_candidates_2d(points, vehicle_band, eps=db_eps, min_points=db_min_points)

    filtered = keep_vehicle_like_reject_walls(
        points=points,
        labels=labels,
        idx=idx,
        h_min=h_min,
        h_max=h_max,
        wall_min_len=wall_min_len,
        wall_max_thickness=wall_max_thickness,
        wall_min_len_over_thickness=wall_min_len_over_thickness,
        veh_max_len=veh_max_len,
        veh_max_w=veh_max_w,
        veh_max_h=veh_max_h,
        veh_min_points=veh_min_points,
    )

    print(f"task2: filtered points: {int(filtered.sum())} / {points.shape[0]}")
    return filtered


def build_world_T_cam(nusc: NuScenes, cam_data_token: str):
    cam_data = nusc.get("sample_data", cam_data_token)
    cam_calib = nusc.get("calibrated_sensor", cam_data["calibrated_sensor_token"])
    ego_pose = nusc.get("ego_pose", cam_data["ego_pose_token"])

    vTc = np.eye(4, dtype=np.float32)
    vTc[:3, :3] = Quaternion(cam_calib["rotation"]).rotation_matrix.astype(np.float32)
    vTc[:3, 3] = np.array(cam_calib["translation"], dtype=np.float32)

    wTv = np.eye(4, dtype=np.float32)
    wTv[:3, :3] = Quaternion(ego_pose["rotation"]).rotation_matrix.astype(np.float32)
    wTv[:3, 3] = np.array(ego_pose["translation"], dtype=np.float32)

    wTc = wTv @ vTc
    K = np.array(cam_calib["camera_intrinsic"], dtype=np.float32)

    return wTc, K


def project_points_to_image(pc_world: np.ndarray, K: np.ndarray, world_T_cam: np.ndarray):
    cam_T_world = np.linalg.inv(world_T_cam)

    pc_h = np.hstack((pc_world, np.ones((pc_world.shape[0], 1), dtype=np.float32)))
    pc_cam = (cam_T_world @ pc_h.T).T[:, :3]

    depths = pc_cam[:, 2].astype(np.float32)

    uvw = (K @ pc_cam.T).T
    uv = uvw[:, :2] / np.maximum(uvw[:, 2:3], 1e-6)

    return uv.astype(np.float32), depths


def sample_image_colors(uv: np.ndarray, depths: np.ndarray, image: np.ndarray):
    h, w = image.shape[0], image.shape[1]

    u = np.round(uv[:, 0]).astype(np.int32)
    v = np.round(uv[:, 1]).astype(np.int32)

    valid = (depths > 0.0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)

    colors = np.zeros((uv.shape[0], 3), dtype=np.float32)
    if np.any(valid):
        pix = image[v[valid], u[valid], :3].astype(np.float32)
        if pix.max() > 1.0:
            pix = pix / 255.0
        colors[valid] = pix

    return colors, valid


def colorize_points_with_cameras(points_world: np.ndarray, cam_data_tokens: list, nusc: NuScenes):
    colors = np.zeros((points_world.shape[0], 3), dtype=np.float32)
    best_dist = np.full((points_world.shape[0],), np.inf, dtype=np.float32)

    for cam_token in cam_data_tokens:
        img_path = nusc.get_sample_data_path(cam_token)
        image = plt.imread(img_path)

        wTc, K = build_world_T_cam(nusc, cam_token)
        uv, depths = project_points_to_image(points_world, K, wTc)
        cam_colors, valid = sample_image_colors(uv, depths, image)

        cam_pos = wTc[:3, 3]
        d = np.linalg.norm(points_world - cam_pos[None, :], axis=1).astype(np.float32)

        update = valid & (d < best_dist) & (cam_colors != 0.0).any(axis=1)
        if np.any(update):
            colors[update] = cam_colors[update]
            best_dist[update] = d[update]

    return colors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="scene-0103")
    parser.add_argument("--max-frames", type=int, default=40)

    # aggregation
    parser.add_argument("--per-frame-voxel", type=float, default=0.20)

    # task2 filter knobs
    parser.add_argument("--ground-cell", type=float, default=0.4)
    parser.add_argument("--ground-percentile", type=float, default=12.0)
    parser.add_argument("--h-min", type=float, default=0.25)
    parser.add_argument("--h-max", type=float, default=3.2)

    parser.add_argument("--db-eps", type=float, default=0.65)
    parser.add_argument("--db-min-points", type=int, default=8)

    parser.add_argument("--wall-min-len", type=float, default=6.0)
    parser.add_argument("--wall-max-thickness", type=float, default=0.35)
    parser.add_argument("--wall-min-len-over-thickness", type=float, default=18.0)

    parser.add_argument("--veh-max-len", type=float, default=15.0)
    parser.add_argument("--veh-max-w", type=float, default=5.0)
    parser.add_argument("--veh-max-h", type=float, default=3.2)
    parser.add_argument("--veh-min-points", type=int, default=25)

    # io
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument("--no-vis", action="store_true")
    parser.add_argument("--export-full-black", action="store_true")

    args = parser.parse_args()

    nusc = NuScenes(version="v1.0-mini", dataroot=str(DATA_ROOT), verbose=False)

    # 1) aggregate
    points, sample_tokens = get_aggregated_cloud_per_frame_voxel(
        nusc=nusc,
        scene_name=args.scene,
        max_frames=args.max_frames,
        per_frame_voxel=args.per_frame_voxel,
    )
    if points.shape[0] == 0:
        raise RuntimeError("no points loaded")

    # 2) filter mask from task2
    filtered_mask = task2_filter_mask(
        points=points,
        ground_cell=args.ground_cell,
        ground_percentile=args.ground_percentile,
        h_min=args.h_min,
        h_max=args.h_max,
        db_eps=args.db_eps,
        db_min_points=args.db_min_points,
        wall_min_len=args.wall_min_len,
        wall_max_thickness=args.wall_max_thickness,
        wall_min_len_over_thickness=args.wall_min_len_over_thickness,
        veh_max_len=args.veh_max_len,
        veh_max_w=args.veh_max_w,
        veh_max_h=args.veh_max_h,
        veh_min_points=args.veh_min_points,
    )
    static_mask = ~filtered_mask

    # 3) cameras from first sample (same baseline style as your friend)
    first_sample = nusc.get("sample", sample_tokens[0])
    cam_names = [
        "CAM_FRONT",
        "CAM_FRONT_LEFT",
        "CAM_FRONT_RIGHT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_BACK_RIGHT",
    ]
    cam_tokens = [first_sample["data"][c] for c in cam_names]

    # 4) colorize ALL points first
    print("task3: colorizing points from 6 cameras")
    colors = colorize_points_with_cameras(points, cam_tokens, nusc)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 5) export static only (filtered removed entirely)
    pcd_static = o3d.geometry.PointCloud()
    pcd_static.points = o3d.utility.Vector3dVector(points[static_mask])
    pcd_static.colors = o3d.utility.Vector3dVector(colors[static_mask])
    static_path = out_dir / f"task3_{args.scene}_colored_static.pcd"
    o3d.io.write_point_cloud(str(static_path), pcd_static)
    print(f"saved: {static_path}")

    # optional: full cloud where filtered points are black
    if args.export_full_black:
        colors_full = colors.copy()
        colors_full[filtered_mask] = 0.0

        pcd_full = o3d.geometry.PointCloud()
        pcd_full.points = o3d.utility.Vector3dVector(points)
        pcd_full.colors = o3d.utility.Vector3dVector(colors_full)

        full_path = out_dir / f"task3_{args.scene}_colored_full_black.pcd"
        o3d.io.write_point_cloud(str(full_path), pcd_full)
        print(f"saved: {full_path}")

    if args.no_vis:
        return

    o3d.visualization.draw_geometries(
        [pcd_static],
        window_name="task3: colored static map (filtered points removed)",
    )


if __name__ == "__main__":
    main()
