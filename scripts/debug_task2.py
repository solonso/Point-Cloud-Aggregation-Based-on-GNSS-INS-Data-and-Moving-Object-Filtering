#!/usr/bin/env python3
"""
Task 2 Debugger (Safe Mode): "Where did my points go?"
Includes Memory Protection (Downsampling) to prevent 'Killed' errors.
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
OUT_DIR = REPO_ROOT / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def get_aggregated_cloud(nusc, scene_name, max_frames):
    scene = next((s for s in nusc.scene if s["name"] == scene_name), None)
    points_list = []
    frame_ids_list = []
    sample_token = scene["first_sample_token"]
    frame_idx = 0
    
    print(f"Aggregating {max_frames} frames...")
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
        
        xyz = pc.points[:3, :].T
        points_list.append(xyz)
        frame_ids_list.append(np.full(xyz.shape[0], frame_idx, dtype=np.int32))
        
        frame_idx += 1
        if max_frames and frame_idx >= max_frames: break
        sample_token = sample["next"]
        
    return np.vstack(points_list), np.concatenate(frame_ids_list)

def analyze_clusters(points, frame_ids):
    # 1. Temporal Filter
    # Using 0.5m (Matches our V3 proposal) to reduce initial load
    print("Step 1: Temporal Filter (Voxel 0.5m)...")
    voxel_size = 0.5
    voxels = np.floor(points / voxel_size).astype(np.int64)
    data = np.hstack([voxels, frame_ids.reshape(-1, 1)])
    unique_voxel_frames = np.unique(data, axis=0)
    unique_voxels, counts = np.unique(unique_voxel_frames[:, :3], axis=0, return_counts=True)
    
    # Static if seen in >= 2 frames (Matches V3)
    static_coords = unique_voxels[counts >= 2]
    static_set = set(map(tuple, static_coords))
    is_static = np.array([tuple(v) in static_set for v in voxels])
    
    print(f"  -> Static: {np.sum(is_static)} | Moving Candidates: {np.sum(~is_static)}")
    
    # 2. Cluster Analysis
    moving_indices = np.where(~is_static)[0]
    moving_points = points[moving_indices]
    
    if len(moving_points) == 0:
        print("No moving candidates found!")
        return

    # --- MEMORY PROTECTION ---
    print("  -> Downsampling candidates to prevent crash...")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(moving_points)
    
    # Downsample: 10cm voxel. keeps shape, reduces count by ~80%
    pcd_down = pcd.voxel_down_sample(voxel_size=0.1) 
    down_points = np.asarray(pcd_down.points)
    print(f"  -> Reduced from {len(moving_points)} to {len(down_points)} points.")

    print("Step 2: Clustering Candidates (DBSCAN)...")
    # eps=1.2 matches V3 proposal
    labels = np.array(pcd_down.cluster_dbscan(eps=1.2, min_points=5, print_progress=True))
    max_label = labels.max()
    print(f"  -> Found {max_label + 1} clusters.")
    
    # CATEGORIZE CLUSTERS
    ground_pts = []
    wall_pts = []
    valid_pts = []
    
    for i in range(max_label + 1):
        mask = (labels == i)
        cluster_pts = down_points[mask]
        
        min_b = cluster_pts.min(axis=0)
        max_b = cluster_pts.max(axis=0)
        dims = max_b - min_b
        
        # --- THE LOGIC (Matches V3) ---
        if dims[2] < 0.2: 
            ground_pts.append(cluster_pts) # Too flat
        elif dims[2] > 2.5 or dims[0] > 35.0:
            wall_pts.append(cluster_pts)   # Too tall or huge
        else:
            valid_pts.append(cluster_pts)  # Valid

    print(f"  -> Classification Results:")
    print(f"     Deleted as GROUND: {sum(len(p) for p in ground_pts)} points")
    print(f"     Deleted as WALLS:  {sum(len(p) for p in wall_pts)} points")
    print(f"     Kept as VALID:     {sum(len(p) for p in valid_pts)} points")

    # SAVE DEBUG FILES
    print("Saving debug PCD files to 'output/'...")
    
    def save_list(name, pt_list, color):
        if not pt_list: return
        p = o3d.geometry.PointCloud()
        all_pts = np.vstack(pt_list)
        p.points = o3d.utility.Vector3dVector(all_pts)
        p.paint_uniform_color(color)
        o3d.io.write_point_cloud(str(OUT_DIR / name), p)

    save_list("debug_ground.pcd", ground_pts, [0, 0, 1]) # BLUE
    save_list("debug_walls.pcd", wall_pts, [0, 1, 0])    # GREEN
    save_list("debug_valid.pcd", valid_pts, [1, 0, 0])   # RED

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="scene-0103")
    parser.add_argument("--max-frames", type=int, default=150)
    args = parser.parse_args()
    
    nusc = NuScenes(version="v1.0-mini", dataroot=str(DATA_ROOT), verbose=False)
    points, frame_ids = get_aggregated_cloud(nusc, args.scene, args.max_frames)
    analyze_clusters(points, frame_ids)

if __name__ == "__main__":
    main()