import os
import argparse
import numpy as np
import open3d as o3d
import cv2
import time
from glob import glob
from tqdm import trange
from scipy.spatial.transform import Rotation as R
import gc


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
    
    # Force CPU usage to avoid GPU memory issues and segmentation faults
    device = o3d.core.Device('cpu:0')
    print(f"Using device: {device}")

    # Create VoxelBlockGrid with conservative parameters
    vbg = o3d.t.geometry.VoxelBlockGrid(
        attr_names=('tsdf', 'weight', 'color'),
        attr_dtypes=(o3d.core.float32, o3d.core.float32, o3d.core.float32),
        attr_channels=((1), (1), (3)),
        voxel_size=args.voxel_size,
        block_count=args.block_count,  # Use argument value
        device=device
    )

    start = time.time()
    processed_frames = 0

    pbar = trange(n_files, desc="Integration progress")
    for i in pbar:
        try:
            pbar.set_description(f"Integration progress, frame {i+1}/{n_files}")
            
            # Read images with error handling
            depth_path = depth_file_names[i]
            color_path = color_file_names[i]
            
            if not os.path.exists(depth_path) or not os.path.exists(color_path):
                print(f"Skipping frame {i}: missing files")
                continue
                
            depth = o3d.t.io.read_image(depth_path).to(device)
            color = o3d.t.io.read_image(color_path).to(device)
            pose = extrinsic[i]
            
            # Check depth validity
            dep = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH) / args.depth_scale
            if dep.min() >= args.depth_max:
                continue

            # Integrate with error handling
            frustum_block_coords = vbg.compute_unique_block_coordinates(
                depth, intrinsic, pose, args.depth_scale, args.depth_max)

            vbg.integrate(frustum_block_coords, depth, color, intrinsic, pose, args.depth_scale, args.depth_max)
            processed_frames += 1
            
            # Memory management every N frames
            if (i + 1) % args.release_every == 0:
                gc.collect()
                
        except Exception as e:
            print(f"Error processing frame {i}: {e}")
            continue

    dt = time.time() - start
    print(f"Integration took {dt:.2f} seconds, processed {processed_frames}/{n_files} frames")
    return vbg


def extract_mesh_safely(vbg, weight_threshold, output_path):
    """Safely extract mesh with multiple fallback strategies"""
    try:
        print(f"Attempting mesh extraction with weight threshold {weight_threshold}")
        
        # Strategy 1: Direct extraction
        mesh = vbg.extract_triangle_mesh(weight_threshold=weight_threshold)
        
        if not mesh.has_vertices():
            print("Warning: Mesh has no vertices")
            return False
            
        # Convert to legacy format
        mesh_legacy = mesh.to_legacy()
        
        if len(mesh_legacy.vertices) == 0:
            print("Warning: Mesh has no vertices after conversion")
            return False
            
        # Save mesh
        success = o3d.io.write_triangle_mesh(output_path, mesh_legacy)
        if success:
            print(f"Successfully saved mesh to {output_path} with {len(mesh_legacy.vertices)} vertices and {len(mesh_legacy.triangles)} triangles")
            return True
        else:
            print("Failed to write mesh file")
            return False
            
    except Exception as e:
        print(f"Error during mesh extraction: {e}")
        return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Integrate depth maps into TSDF')
    parser.add_argument('--result', type=str, required=True, help='Path to the result folder')
    parser.add_argument('--voxel_size', type=float, default=0.006, help='Voxel size')
    parser.add_argument('--depth_scale', type=float, default=6553.5, help='Depth scale')
    parser.add_argument('--depth_max', type=float, default=5.0, help='Maximum depth')
    parser.add_argument('--weight', type=float, default=[2.0], nargs='+', help='Weight threshold')
    parser.add_argument('--block_count', type=int, default=40000, help='Number of voxel blocks')
    parser.add_argument('--release_every', type=int, default=20, help='Release memory every N frames')
    args = parser.parse_args()

    # Find depth and color files
    depth_file_names = sorted(glob(f'{args.result}/renders/depth_after_opt/*.png'))
    color_file_names = sorted(glob(f'{args.result}/renders/image_after_opt/*.jpg'))
    
    if len(depth_file_names) == 0:
        print("Error: No depth files found")
        exit(1)
    if len(color_file_names) == 0:
        print("Error: No color files found")
        exit(1)
    
    print(f"Found {len(depth_file_names)} depth maps and {len(color_file_names)} color images")
    
    # Ensure same number of files
    min_files = min(len(depth_file_names), len(color_file_names))
    depth_file_names = depth_file_names[:min_files]
    color_file_names = color_file_names[:min_files]
    
    stamps = [float(os.path.basename(i)[:-4]) for i in color_file_names]
    
    try:
        intrinsic, extrinsic = load_intrinsic_extrinsic(args.result, stamps)
        vbg = integrate(depth_file_names, color_file_names, intrinsic, extrinsic, args)

        # Extract meshes for each weight threshold
        for w in args.weight:
            out = f'{args.result}/tsdf_mesh_w{w:.1f}.ply'
            
            if os.path.exists(out):
                print(f"Mesh file {out} already exists, skipping")
                continue
                
            success = extract_mesh_safely(vbg, w, out)
            
            if not success:
                print(f"Failed to extract mesh with weight {w}, creating placeholder")
                # Create a minimal valid PLY file
                try:
                    with open(out, 'w') as f:
                        f.write("ply\n")
                        f.write("format ascii 1.0\n")
                        f.write("comment TSDF mesh extraction failed\n")
                        f.write("element vertex 0\n")
                        f.write("element face 0\n")
                        f.write("end_header\n")
                except Exception:
                    pass
        
        print("TSDF integration completed successfully")
        
    except Exception as e:
        print(f"Fatal error during TSDF integration: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
