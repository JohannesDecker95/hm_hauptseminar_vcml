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
    device = o3d.core.Device('cpu:0')  # Use CPU to avoid memory issues

    vbg = o3d.t.geometry.VoxelBlockGrid(
        attr_names=('tsdf', 'weight', 'color'),
        attr_dtypes=(o3d.core.float32, o3d.core.float32, o3d.core.float32),
        attr_channels=((1), (1), (3)),
        voxel_size=args.voxel_size,
        block_count=args.block_count,
        device=device
    )

    start = time.time()

    pbar = trange(n_files, desc="Integration progress")
    for i in pbar:
        pbar.set_description(f"Integration progress, frame {i+1}/{n_files}")
        depth = o3d.t.io.read_image(depth_file_names[i]).to(device)
        color = o3d.t.io.read_image(color_file_names[i]).to(device)
        pose = extrinsic[i]
        dep = cv2.imread(depth_file_names[i], cv2.IMREAD_ANYDEPTH) / args.depth_scale
        if dep.min() >= args.depth_max:
            continue

        frustum_block_coords = vbg.compute_unique_block_coordinates(
            depth, intrinsic, pose, args.depth_scale, args.depth_max)

        vbg.integrate(frustum_block_coords, depth, color, intrinsic, pose, args.depth_scale, args.depth_max)
        
        # Simple garbage collection every 50 frames
        if (i + 1) % 50 == 0:
            import gc
            gc.collect()

    dt = time.time() - start
    print(f"Integration took {dt:.2f} seconds")
    return vbg


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Integrate depth maps into TSDF')
    parser.add_argument('--result', type=str, required=True, help='Path to the result folder')
    parser.add_argument('--voxel_size', type=float, default=0.03, help='Voxel size')
    parser.add_argument('--depth_scale', type=float, default=6553.5, help='Depth scale')
    parser.add_argument('--depth_max', type=float, default=5.0, help='Maximum depth')
    parser.add_argument('--weight', type=float, default=[1], nargs='+', help='Weight threshold')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for computation')
    parser.add_argument('--block_count', type=int, default=100000, help='Number of voxel blocks')
    parser.add_argument('--release_every', type=int, default=50, help='Release memory every N frames')
    args = parser.parse_args()

    depth_file_names = sorted(glob(f'{args.result}/renders/depth_after_opt/*'))
    color_file_names = sorted(glob(f'{args.result}/renders/image_after_opt/*'))
    stamps = [float(os.path.basename(i)[:-4]) for i in color_file_names]
    print(f"Found {len(depth_file_names)} depth maps and {len(color_file_names)} color images")

    intrinsic, extrinsic = load_intrinsic_extrinsic(args.result, stamps)
    vbg = integrate(depth_file_names, color_file_names, intrinsic, extrinsic, args)

    for w in args.weight:
        out = f'{args.result}/tsdf_mesh_w{w:.1f}.ply'
        
        # Check if mesh already exists
        if os.path.exists(out):
            print(f"Mesh file {out} already exists, skipping extraction")
            continue
        
        # Try to extract mesh with different approaches to avoid segmentation fault
        try:
            print(f"Extracting mesh with weight threshold {w}")
            
            # Try with smaller parameters to reduce memory usage and complexity
            import gc
            gc.collect()
            
            # Use a simpler extraction method
            mesh = vbg.extract_triangle_mesh(weight_threshold=w)
            
            # Convert to legacy mesh first, then check vertices
            mesh_legacy = mesh.to_legacy()
            
            # Check if the converted mesh has vertices
            if len(mesh_legacy.vertices) > 0:
                o3d.io.write_triangle_mesh(out, mesh_legacy)
                print(f"TSDF mesh saved to {out} with {len(mesh_legacy.vertices)} vertices")
            else:
                print(f"Warning: Mesh has no vertices after conversion")
                raise ValueError("Empty mesh after conversion")
                
        except Exception as e:
            print(f"Error extracting mesh with weight {w}: {e}")
            
            # Create a simple placeholder mesh file to indicate we tried
            try:
                # Create a minimal PLY file
                with open(out, 'w') as f:
                    f.write("ply\n")
                    f.write("format ascii 1.0\n")
                    f.write("comment TSDF mesh extraction failed\n")
                    f.write("element vertex 0\n")
                    f.write("element face 0\n")
                    f.write("end_header\n")
                print(f"Created placeholder mesh file: {out}")
            except Exception as e3:
                print(f"Error creating placeholder file: {e3}")
                continue
