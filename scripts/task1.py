#!/usr/bin/env python3
"""
Name: Nwafor Solomon Chibuzo
Nuptun ID: UITK8W
Task 1: Aggregate nuScenes LIDAR_TOP frames into the global frame and visualize.

The nuScenes devkit transforms
(`LidarPointCloud` + `transform_matrix`) and show an Open3D window. No
annotations are used here.

Run from repo root:
    python scripts/task1.py --scene scene-0103 --max-frames 40
"""

import argparse
from pathlib import Path
from typing import Optional  # kept for consistency even if unused

import numpy as np


# repo root = parent of /scripts
# used to build stable relative paths for dataset and outputs
REPO_ROOT = Path(__file__).resolve().parents[1]

# expected nuScenes folder inside the repo root
# this matches the assignment packaging style (zip contains nuscenes/)
DATA_ROOT = REPO_ROOT / "nuscenes"


def main():
    # cli args allow quick scene switching and reproducible runs
    parser = argparse.ArgumentParser(description="Aggregate and visualize LIDAR_TOP frames.")

    # scene name in nuScenes scene.json, eg scene-0103 in v1.0-mini
    parser.add_argument("--scene", default="scene-0103", help="Scene name from scene.json (default: scene-0103)")

    # optional cap for faster debugging and short demo runs
    parser.add_argument("--max-frames", type=int, default=None, help="Optional frame limit for quick tests")

    # disable visualization, useful on headless machines or for batch runs
    parser.add_argument("--no-view", action="store_true", help="Skip Open3D visualization")

    # choose how to color points in Open3D
    # height: map z values to color
    # intensity: grayscale from lidar intensity channel
    # gray: constant color for all points
    parser.add_argument(
        "--color",
        choices=["gray", "height", "intensity"],
        default="height",
        help="Color scheme for visualization",
    )

    # open3d render option: point size on screen
    parser.add_argument("--point-size", type=float, default=1.5, help="Open3D point size")

    # cli flag requested by assignment style
    # current script always saves anyway, but we keep the flag for interface completeness
    parser.add_argument(
        "--save-pcd",
        action="store_true",
        help="Also save aggregated cloud to output/<scene>.pcd with colors",
    )

    args = parser.parse_args()

    # import heavy dependencies here so the script can fail with a clear message
    # also avoids import overhead when someone only wants to inspect the file
    try:
        from nuscenes.nuscenes import NuScenes
        from nuscenes.utils.data_classes import LidarPointCloud
        from nuscenes.utils.geometry_utils import transform_matrix
        from pyquaternion import Quaternion
    except Exception as exc:  # noqa: BLE001
        # exit cleanly with actionable install hint
        raise SystemExit(
            "nuScenes devkit (nuscenes-devkit) and pyquaternion are required for this script. "
            f"Error: {exc}"
        ) from exc

    # load nuScenes metadata (scenes, samples, calibrated sensors, ego poses)
    # v1.0-mini keeps runtime reasonable for assignment submission
    nusc = NuScenes(version="v1.0-mini", dataroot=str(DATA_ROOT), verbose=False)

    # find the requested scene by name
    scene = next((s for s in nusc.scene if s["name"] == args.scene), None)
    if scene is None:
        raise SystemExit(f"Scene {args.scene} not found.")

    # frames will store per-frame xyz arrays in global frame
    # intensities will store lidar intensity per point for optional coloring
    frames = []
    intensities = []
    num_points = 0

    # nuScenes stores a linked list of samples: first_sample_token -> next -> next ...
    sample_token = scene["first_sample_token"]
    frame_idx = 0

    # iterate over all samples in the scene (or stop early with --max-frames)
    while sample_token:
        # sample contains sensor data tokens for this timestamp
        sample = nusc.get("sample", sample_token)

        # we use only top lidar for this task
        lidar_token = sample["data"]["LIDAR_TOP"]

        # sample_data holds file path info and tokens to calibration and ego pose
        sd = nusc.get("sample_data", lidar_token)

        # calibrated_sensor provides sensor extrinsics relative to ego frame
        calib = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])

        # ego_pose provides GNSS-INS pose of the vehicle in global frame at this timestamp
        ego = nusc.get("ego_pose", sd["ego_pose_token"])

        # load raw lidar point cloud from disk using nuScenes devkit loader
        # points are stored in a 4xN array: x,y,z,intensity (in sensor frame)
        pc = LidarPointCloud.from_file(nusc.get_sample_data_path(lidar_token))

        # build 4x4 homogeneous transform sensor -> ego
        # calib["rotation"] is quaternion, calib["translation"] is xyz
        sensor_to_ego = transform_matrix(calib["translation"], Quaternion(calib["rotation"]), inverse=False)

        # build 4x4 homogeneous transform ego -> global
        # ego pose also given as quaternion + translation
        ego_to_global = transform_matrix(ego["translation"], Quaternion(ego["rotation"]), inverse=False)

        # transform the cloud into global frame:
        # P_global = T_ego_to_global * T_sensor_to_ego * P_sensor
        pc.transform(ego_to_global @ sensor_to_ego)

        # extract xyz points as Nx3 for numpy stacking
        xyz = pc.points[:3, :].T

        # extract intensity channel (1D length N)
        intens = pc.points[3, :]

        # accumulate per-frame arrays (we stack at the end for efficiency)
        frames.append(xyz)
        intensities.append(intens)
        num_points += xyz.shape[0]
        frame_idx += 1

        # lightweight progress logging for long scenes
        if frame_idx % 10 == 0:
            print(f"[{args.scene}] processed {frame_idx} frames ({num_points} pts)")

        # stop early if user requested a cap
        if args.max_frames and frame_idx >= args.max_frames:
            break

        # move to the next sample in the linked list
        sample_token = sample["next"]

    # final aggregation stats for reproducibility in the report
    print(f"[{args.scene}] done. Frames: {len(frames)}, points: {num_points}")
    if not frames:
        raise SystemExit("No frames aggregated.")

    # build one large Nx3 global array from all frames
    cloud = np.vstack(frames)

    # build one long intensity vector aligned with cloud
    intensity = np.concatenate(intensities)

    # allow non-visual runs for grading scripts or remote machines
    if args.no_view:
        return

    # import open3d only when needed
    try:
        import open3d as o3d
    except ImportError:
        # still keep the run useful: user can rerun with --no-view
        print("Open3D not installed; install open3d to visualize, or rerun with --no-view.")
        return

    # Choose colors
    if args.color == "height":
        # color by height (z) with robust scaling to avoid outliers dominating the range
        z = cloud[:, 2]
        z_min, z_max = np.percentile(z, [1, 99])  # robust range
        span = max(z_max - z_min, 1e-3)
        norm = np.clip((z - z_min) / span, 0, 1)

        def hsv_to_rgb(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
            """Vectorized HSV->RGB for 0-1 inputs."""
            # map hue to one of 6 sectors, then compute linear interpolation inside sector
            i = np.floor(h * 6).astype(int)
            f = h * 6 - i
            p = v * (1 - s)
            q = v * (1 - f * s)
            t = v * (1 - (1 - f) * s)
            i_mod = i % 6
            r = np.choose(i_mod, [v, q, p, p, t, v])
            g = np.choose(i_mod, [t, v, v, q, p, p])
            b = np.choose(i_mod, [p, p, t, v, v, q])
            return np.stack([r, g, b], axis=-1)

        # hue mapping chosen for visual separation across height
        hues = 0.66 * (1 - norm)  # blue high, red low
        colors = hsv_to_rgb(hues, np.ones_like(hues) * 0.8, np.ones_like(hues) * 0.9)

    elif args.color == "intensity":
        # grayscale by intensity with robust scaling
        lo, hi = np.percentile(intensity, [1, 99])
        span = max(hi - lo, 1e-3)
        norm = np.clip((intensity - lo) / span, 0, 1)
        colors = np.stack([norm, norm, norm], axis=-1)

    else:  # gray
        # constant gray for all points
        colors = np.full((cloud.shape[0], 3), 0.85)

    # create open3d point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # ensure output directory exists
    out_dir = REPO_ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # output naming kept consistent with existing repository outputs
    out_path = out_dir / f"taask1_{args.scene}.pcd"

    # save PCD for submission and for later tasks
    o3d.io.write_point_cloud(str(out_path), pcd)
    print(f"Saved PCD to {out_path}")

    # open3d visualizer gives interactive rotate/zoom for demo video recording
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"Aggregated {args.scene}")
    vis.add_geometry(pcd)

    # set point size for visibility in recordings
    ropt = vis.get_render_option()
    ropt.point_size = args.point_size

    # run event loop until user closes the window
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
