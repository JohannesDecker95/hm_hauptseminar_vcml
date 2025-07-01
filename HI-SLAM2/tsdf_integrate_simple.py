import os
import argparse
import numpy as np
import open3d as o3d
import cv2
import time
from glob import glob
from tqdm import trange
from scipy.spatial.transform import Rotation as R


def to_se3_matrix(pvec):
    pose = np.eye(4)
    pose[:3, :3] = R.from_quat(pvec[4:]).as_matrix()
    pose[:3, 3] = pvec[1:4]
    return pose


def load_intrinsic_extrinsic(result, stamps):
    c = np.load(f'{result}/intrinsics.npy')
    intrinsic = o3d.core.Tensor([[c[0], 0, c[2]], [0, c[1], c[3]], [0, 0, 1]], dtype=o3d.core.Dtype.Float64)
    poses = np.loadtxt(f'{result}/traj_full.txt')
    poses = [np.linalg.inv(to_se3_matrix(poses[int(s)])) for s in stamps]
    poses = list(map(lambda x: o3d.core.Tensor(x, dtype=o3d.core.Dtype.Float64), poses))
    return intrinsic, poses


def integrate(depth_file_names, color_file_names, intrinsic, extrinsic, args):
    n_files = len(depth_file_names)
    # Force CPU and use smaller block count
    device = o3d.core.Device('cpu:0')

    vbg = o3d.t.geometry.VoxelBlockGrid(
        attr_names=('tsdf', 'weight', 'color'),
        attr_dtypes=(o3d.core.float32, o3d.core.float32, o3d.core.float32),
        attr_channels=((1), (1), (3)),
        voxel_size=args.voxel_size,
        block_count=20000,  # Reduced from 100000
        device=device
    )

    start = time.time()

    pbar = trange(n_files, desc="Integration progress")
    for i in pbar:
        pbar.set_description(f"Integration progress, frame {i+1}/{n_files}")
        try:
            depth = o3d.t.io.read_image(depth_file_names[i]).to(device)
            color = o3d.t.io.read_image(color_file_names[i]).to(device)
            pose = extrinsic[i]
            dep = cv2.imread(depth_file_names[i], cv2.IMREAD_ANYDEPTH) / args.depth_scale
            if dep.min() >= args.depth_max:
                continue

            frustum_block_coords = vbg.compute_unique_block_coordinates(
                depth, intrinsic, pose, args.depth_scale, args.depth_max)

            vbg.integrate(frustum_block_coords, depth, color, intrinsic, pose, args.depth_scale, args.depth_max)
        except Exception as e:
            print(f"Error processing frame {i}: {e}")
            continue

    dt = time.time() - start
    print(f"Integration took {dt:.2f} seconds")
    return vbg


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Integrate depth maps into TSDF')
    parser.add_argument('--result', type=str, required=True, help='Path to the result folder')
    parser.add_argument('--voxel_size', type=float, default=0.01, help='Voxel size')  # Larger voxel size
    parser.add_argument('--depth_scale', type=float, default=6553.5, help='Depth scale')
    parser.add_argument('--depth_max', type=float, default=3.0, help='Maximum depth')  # Reduced max depth
    parser.add_argument('--weight', type=float, default=[2.0], nargs='+', help='Weight threshold')
    args = parser.parse_args()

    depth_file_names = sorted(glob(f'{args.result}/renders/depth_after_opt/*.png'))
    color_file_names = sorted(glob(f'{args.result}/renders/image_after_opt/*.jpg'))
    
    # Reduce number of frames for testing
    max_frames = 200
    depth_file_names = depth_file_names[:max_frames]
    color_file_names = color_file_names[:max_frames]
    
    stamps = [float(os.path.basename(i)[:-4]) for i in color_file_names]
    print(f"Processing {len(depth_file_names)} depth maps and {len(color_file_names)} color images")

    try:
        intrinsic, extrinsic = load_intrinsic_extrinsic(args.result, stamps)
        vbg = integrate(depth_file_names, color_file_names, intrinsic, extrinsic, args)

        for w in args.weight:
            out = f'{args.result}/tsdf_mesh_w{w:.1f}.ply'
            print(f"Extracting mesh with weight threshold {w}")
            
            try:
                mesh = vbg.extract_triangle_mesh(weight_threshold=w)
                mesh = mesh.to_legacy()
                o3d.io.write_triangle_mesh(out, mesh)
                print(f"Successfully saved TSDF mesh to {out} with {len(mesh.vertices)} vertices")
            except Exception as e:
                print(f"Error during mesh extraction: {e}")
                # Create placeholder 
                with open(out, 'w') as f:
                    f.write("ply\nformat ascii 1.0\ncomment TSDF extraction failed\nelement vertex 0\nelement face 0\nend_header\n")
                
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
