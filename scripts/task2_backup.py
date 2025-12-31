#!/usr/bin/env python3
"""
Task 2: Moving Object Filtering via Temporal Consistency.

Algorithm:
1. Aggregate all points into a Global Frame (same as Task 1).
2. Discretize space into a Voxel Grid (3D cubes).
3. For each voxel, count the number of UNIQUE frames that contributed points to it.
   - If a voxel contains points from many different timestamps, it is STATIC.
   - If a voxel contains points from only a few timestamps, it is likely MOVING (transient).
4. Filter the cloud based on this 'frame_count' threshold.

Comparison to Friends' implementation:
- Friends: Used 'annotations' (Cheat/Illegal).
- Us: Use 'geometry + time' (Valid/Robust).
"""

import argparse
from pathlib import Path
import numpy as np

# Try importing required libraries
try:
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.data_classes import LidarPointCloud
    from nuscenes.utils.geometry_utils import transform_matrix
    from pyquaternion import Quaternion
    import open3d as o3d
except ImportError as e:
    raise SystemExit(f"Missing required library: {e}. Please install nuscenes-devkit and open3d.")

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "nuscenes"

def get_aggregated_cloud_with_frame_ids(nusc, scene_name, max_frames=None):
    """
    Aggregates LiDAR points into global frame, tracking which frame they came from.
    
    Returns:
        all_points (np.array): Nx3 array of (x, y, z) global coordinates.
        all_frame_ids (np.array): N array of frame indices (0, 1, 2...) corresponding to each point.
    """
    scene = next((s for s in nusc.scene if s["name"] == scene_name), None)
    if not scene:
        raise ValueError(f"Scene {scene_name} not found.")

    points_list = []
    frame_ids_list = []
    
    sample_token = scene["first_sample_token"]
    frame_idx = 0

    print(f"[{scene_name}] Aggregating frames...")

    while sample_token:
        # 1. Get sample and sensor data
        sample = nusc.get("sample", sample_token)
        lidar_token = sample["data"]["LIDAR_TOP"]
        sd = nusc.get("sample_data", lidar_token)
        calib = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
        ego = nusc.get("ego_pose", sd["ego_pose_token"])

        # 2. Load Point Cloud
        pc = LidarPointCloud.from_file(nusc.get_sample_data_path(lidar_token))

        # 3. Transform to Global Frame
        # Sensor -> Ego
        sensor_to_ego = transform_matrix(calib["translation"], Quaternion(calib["rotation"]), inverse=False)
        # Ego -> Global
        ego_to_global = transform_matrix(ego["translation"], Quaternion(ego["rotation"]), inverse=False)
        # Combine
        pc.transform(ego_to_global @ sensor_to_ego)

        # 4. Store points and their frame index
        xyz = pc.points[:3, :].T  # (N, 3)
        points_list.append(xyz)
        
        # Create an array of the same length filled with the current frame_idx
        frame_ids = np.full(xyz.shape[0], frame_idx, dtype=np.int32)
        frame_ids_list.append(frame_ids)

        frame_idx += 1
        if frame_idx % 10 == 0:
            print(f"  Processed {frame_idx} frames...")
        if max_frames and frame_idx >= max_frames:
            break

        sample_token = sample["next"]

    print(f"[{scene_name}] Done. Total frames: {frame_idx}")
    return np.vstack(points_list), np.concatenate(frame_ids_list)

def filter_moving_objects(points, frame_ids, voxel_size=0.25, min_frames_seen=15):
    """
    Filters moving objects by checking temporal consistency of voxels.
    
    Args:
        points: (Nx3) Global point cloud.
        frame_ids: (N) Frame index for each point.
        voxel_size: Size of the grid cell in meters.
        min_frames_seen: A voxel must be seen in at least this many unique frames to be 'static'.
    
    Returns:
        static_mask (np.array bool): True for points that are static.
    """
    print(f"\nFiltering moving objects (Voxel: {voxel_size}m, Threshold: {min_frames_seen} frames)...")
    
    # 1. Quantize points to integer voxel coordinates
    # We divide coordinates by voxel_size and floor them to get integer indices
    voxels = np.floor(points / voxel_size).astype(np.int64)

    # 2. Use a dictionary to track unique frames per voxel
    # Key: Tuple (ix, iy, iz), Value: Set of frame_ids
    voxel_history = {}

    # Optimization: To avoid slow Python loops, we use pandas or pure numpy sorting.
    # Here we stick to pure numpy for minimal dependencies.
    
    # Pack 3D coordinates into a single structure to sort/unique easier
    # (We structure them as rows to allow lexicographical sort)
    # However, Python dicts are reasonably fast for sparse hashing. Let's use a loop over unique points? 
    # No, that's too slow for millions of points.
    
    # BETTER APPROACH: Lex-sort by voxel coordinates
    # Combine voxel coords + frame_id
    data = np.hstack([voxels, frame_ids.reshape(-1, 1)]) # Nx4 array: [x, y, z, frame]
    
    # Find unique (voxel, frame) pairs. 
    # This removes multiple points falling in the same voxel in the SAME frame.
    unique_voxel_frames = np.unique(data, axis=0) 
    
    # Now we have a list of [x, y, z, frame]. We need to count how many frames each [x, y, z] has.
    unique_voxels, counts = np.unique(unique_voxel_frames[:, :3], axis=0, return_counts=True)
    
    # 3. Identify Static Voxels
    # These are voxels where 'count' (number of unique frames) >= threshold
    static_voxel_coords = unique_voxels[counts >= min_frames_seen]
    
    print(f"Found {len(static_voxel_coords)} static voxels out of {len(unique_voxels)} total occupied voxels.")
    print(f"Static voxels: {(counts >= min_frames_seen).sum()} / {len(unique_voxels)} voxels")
    print(f"Moving voxels: {(counts < min_frames_seen).sum()} / {len(unique_voxels)} voxels")

    # 4. Map back to original points
    # We put the static voxels into a Python set for O(1) lookup
    # Turning rows into tuples for hashing
    static_set = set(map(tuple, static_voxel_coords))
    
    # Generate the mask
    # We check if each point's voxel coordinate is in the static_set
    print("Mapping voxels back to points...")
    is_static = np.array([tuple(v) in static_set for v in voxels])
    
    n_static = is_static.sum()
    n_moving = (~is_static).sum()
    print(f"\nFinal Classification:")
    print(f"  Static points:  {n_static:,} ({100*n_static/len(points):.1f}%)")
    print(f"  Moving points:  {n_moving:,} ({100*n_moving/len(points):.1f}%)")
    
    return is_static

def visualize_results(points, is_static):
    """
    Visualizes the result: Static points in Gray, Moving points in Red.
    """
    print("\nVisualizing...")
    
    # 1. Create colors array
    # Default Gray [0.5, 0.5, 0.5]
    colors = np.full(points.shape, 0.5)
    
    # Paint Moving points (where is_static is False) RED [1, 0, 0]
    colors[~is_static] = [1, 0, 0]
    
    # Paint Static points slightly darker/different to contrast? Let's keep them gray.
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Task 2: Moving Object Filter (Red = Moving)")
    vis.add_geometry(pcd)
    
    # Set background to black for better contrast
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    opt.point_size = 1.0
    
    vis.run()
    vis.destroy_window()

def save_results(points, is_static, scene_name):
    """
    Save static and moving point clouds to files.
    """
    out_dir = REPO_ROOT / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Save static cloud
    static_pcd = o3d.geometry.PointCloud()
    static_pcd.points = o3d.utility.Vector3dVector(points[is_static])
    static_path = out_dir / f"{scene_name}_static.pcd"
    o3d.io.write_point_cloud(str(static_path), static_pcd)
    print(f"Saved static cloud to {static_path}")
    
    # Save moving cloud
    moving_pcd = o3d.geometry.PointCloud()
    moving_pcd.points = o3d.utility.Vector3dVector(points[~is_static])
    moving_path = out_dir / f"{scene_name}_moving.pcd"
    o3d.io.write_point_cloud(str(moving_path), moving_pcd)
    print(f"Saved moving cloud to {moving_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="scene-0103", help="Scene name")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit frames for testing")
    parser.add_argument("--voxel-size", type=float, default=0.2, help="Voxel size in meters")
    parser.add_argument("--threshold", type=int, default=10, help="Min frames to be considered static")
    parser.add_argument("--no-view", action="store_true", help="Skip visualization")
    parser.add_argument("--save-pcd", action="store_true", help="Save point clouds to files")
    args = parser.parse_args()

    # Initialize NuScenes
    nusc = NuScenes(version="v1.0-mini", dataroot=str(DATA_ROOT), verbose=False)

    # 1. Aggregate
    points, frame_ids = get_aggregated_cloud_with_frame_ids(nusc, args.scene, args.max_frames)
    
    # 2. Filter
    is_static = filter_moving_objects(points, frame_ids, 
                                      voxel_size=args.voxel_size, 
                                      min_frames_seen=args.threshold)

    # 3. Save if requested
    if args.save_pcd:
        save_results(points, is_static, args.scene)

    # 4. Visualize
    if not args.no_view:
        visualize_results(points, is_static)

if __name__ == "__main__":
    main()
