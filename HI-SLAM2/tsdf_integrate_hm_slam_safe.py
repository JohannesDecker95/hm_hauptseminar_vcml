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
import sys
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)


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


def create_point_cloud_mesh(depth_file_names, color_file_names, intrinsic, extrinsic, args):
    """
    Alternative approach: create mesh from accumulated point clouds instead of TSDF
    This avoids the segmentation fault in Open3D's TSDF mesh extraction
    """
    print("Using point cloud accumulation approach to avoid TSDF segmentation faults")
    
    device = o3d.core.Device('cpu:0')
    
    # Convert intrinsic tensor to numpy for camera operations
    intrinsic_np = intrinsic.cpu().numpy()
    
    all_points = []
    all_colors = []
    
    # Sample every N frames to reduce computation
    sample_rate = max(1, len(depth_file_names) // 100)  # Max 100 frames
    indices = list(range(0, len(depth_file_names), sample_rate))
    
    print(f"Processing {len(indices)} frames (sampling every {sample_rate} frames)")
    
    for i in trange(len(indices), desc="Accumulating point clouds"):
        idx = indices[i]
        
        try:
            # Load depth and color
            depth_img = cv2.imread(depth_file_names[idx], cv2.IMREAD_ANYDEPTH)
            color_img = cv2.imread(color_file_names[idx])
            
            if depth_img is None or color_img is None:
                continue
            
            # Convert depth to meters
            depth_img = depth_img.astype(np.float32) / args.depth_scale
            
            # Filter by depth range
            valid_mask = (depth_img > 0.1) & (depth_img < args.depth_max)
            
            if not valid_mask.any():
                continue
            
            # Create point cloud from depth image
            height, width = depth_img.shape
            fx, fy = intrinsic_np[0, 0], intrinsic_np[1, 1]
            cx, cy = intrinsic_np[0, 2], intrinsic_np[1, 2]
            
            # Generate pixel coordinates
            u, v = np.meshgrid(np.arange(width), np.arange(height))
            
            # Apply mask
            u_valid = u[valid_mask]
            v_valid = v[valid_mask]
            depth_valid = depth_img[valid_mask]
            
            # Back-project to 3D
            x = (u_valid - cx) * depth_valid / fx
            y = (v_valid - cy) * depth_valid / fy
            z = depth_valid
            
            # Stack to get 3D points
            points_cam = np.stack([x, y, z], axis=1)
            
            # Transform to world coordinates
            pose = extrinsic[idx].cpu().numpy()
            points_hom = np.hstack([points_cam, np.ones((points_cam.shape[0], 1))])
            points_world = (pose @ points_hom.T).T[:, :3]
            
            # Get colors
            colors = color_img[valid_mask] / 255.0  # Normalize to [0,1]
            colors = colors[:, [2, 1, 0]]  # BGR to RGB
            
            all_points.append(points_world)
            all_colors.append(colors)
            
        except Exception as e:
            print(f"Warning: Failed to process frame {idx}: {e}")
            continue
    
    if not all_points:
        print("Error: No valid point clouds generated")
        return None
    
    # Combine all point clouds
    combined_points = np.vstack(all_points)
    combined_colors = np.vstack(all_colors)
    
    print(f"Generated combined point cloud with {len(combined_points)} points")
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(combined_points)
    pcd.colors = o3d.utility.Vector3dVector(combined_colors)
    
    # Downsample to reduce complexity
    voxel_size = args.voxel_size * 2  # Use larger voxel size for downsampling
    pcd = pcd.voxel_down_sample(voxel_size)
    print(f"After downsampling: {len(pcd.points)} points")
    
    if len(pcd.points) < 1000:
        print("Warning: Too few points after downsampling")
        return None
    
    # Remove outliers
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    print(f"After outlier removal: {len(pcd.points)} points")
    
    return pcd


def create_mesh_from_pointcloud(pcd, output_path):
    """
    Create mesh from point cloud using multiple strategies
    """
    print("Creating mesh from point cloud...")
    
    # Strategy 1: Ball pivoting
    try:
        print("Trying ball pivoting reconstruction...")
        
        # Estimate normals
        pcd.estimate_normals()
        pcd.orient_normals_consistent_tangent_plane(100)
        
        # Ball pivoting
        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        radii = [avg_dist, avg_dist * 2, avg_dist * 4]
        
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector(radii))
        
        if len(mesh.vertices) > 0:
            # Clean mesh
            mesh.remove_duplicated_vertices()
            mesh.remove_degenerate_triangles()
            mesh.remove_unreferenced_vertices()
            
            if len(mesh.vertices) > 0:
                o3d.io.write_triangle_mesh(output_path, mesh)
                print(f"Success: Ball pivoting mesh with {len(mesh.vertices)} vertices")
                return True
        
    except Exception as e:
        print(f"Ball pivoting failed: {e}")
    
    # Strategy 2: Poisson reconstruction
    try:
        print("Trying Poisson reconstruction...")
        
        if not pcd.has_normals():
            pcd.estimate_normals()
        
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=6)
        
        if len(mesh.vertices) > 0:
            # Clean mesh
            mesh.remove_duplicated_vertices()
            mesh.remove_degenerate_triangles()
            mesh.remove_unreferenced_vertices()
            
            if len(mesh.vertices) > 0:
                o3d.io.write_triangle_mesh(output_path, mesh)
                print(f"Success: Poisson mesh with {len(mesh.vertices)} vertices")
                return True
        
    except Exception as e:
        print(f"Poisson reconstruction failed: {e}")
    
    # Strategy 3: Alpha shapes
    try:
        print("Trying alpha shapes...")
        
        # Create alpha shape mesh
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha=0.1)
        
        if len(mesh.vertices) > 0:
            mesh.remove_duplicated_vertices()
            mesh.remove_degenerate_triangles()
            mesh.remove_unreferenced_vertices()
            
            if len(mesh.vertices) > 0:
                o3d.io.write_triangle_mesh(output_path, mesh)
                print(f"Success: Alpha shape mesh with {len(mesh.vertices)} vertices")
                return True
        
    except Exception as e:
        print(f"Alpha shapes failed: {e}")
    
    # Strategy 4: Convex hull as last resort
    try:
        print("Creating convex hull as fallback...")
        
        mesh, _ = pcd.compute_convex_hull()
        
        if len(mesh.vertices) > 0:
            o3d.io.write_triangle_mesh(output_path, mesh)
            print(f"Fallback: Convex hull mesh with {len(mesh.vertices)} vertices")
            return True
        
    except Exception as e:
        print(f"Convex hull failed: {e}")
    
    return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Alternative TSDF-free mesh generation for HM_SLAM')
    parser.add_argument('--result', type=str, required=True, help='Path to the result folder')
    parser.add_argument('--voxel_size', type=float, default=0.006, help='Voxel size for downsampling')
    parser.add_argument('--depth_scale', type=float, default=6553.5, help='Depth scale')
    parser.add_argument('--depth_max', type=float, default=5.0, help='Maximum depth')
    parser.add_argument('--weight', type=float, default=[2.0], nargs='+', help='Weight threshold (unused in this version)')
    args = parser.parse_args()
    
    depth_file_names = sorted(glob(f'{args.result}/renders/depth_after_opt/*'))
    color_file_names = sorted(glob(f'{args.result}/renders/image_after_opt/*'))
    
    if not depth_file_names or not color_file_names:
        print(f"Error: No depth or color files found in {args.result}/renders/")
        sys.exit(1)
    
    stamps = [float(os.path.basename(i)[:-4]) for i in color_file_names]
    print(f"Found {len(depth_file_names)} depth maps and {len(color_file_names)} color images")

    # Validate that we have the required files
    if not os.path.exists(f'{args.result}/intrinsics.npy'):
        print(f"Error: intrinsics.npy not found in {args.result}")
        sys.exit(1)
    
    if not os.path.exists(f'{args.result}/traj_full.txt'):
        print(f"Error: traj_full.txt not found in {args.result}")
        sys.exit(1)

    try:
        intrinsic, extrinsic = load_intrinsic_extrinsic(args.result, stamps)
        print(f"Loaded intrinsics and {len(extrinsic)} poses")
    except Exception as e:
        print(f"Error loading intrinsics/extrinsics: {e}")
        sys.exit(1)

    # Create point cloud mesh instead of TSDF
    try:
        pcd = create_point_cloud_mesh(depth_file_names, color_file_names, intrinsic, extrinsic, args)
        if pcd is None:
            print("Failed to create point cloud")
            sys.exit(1)
    except Exception as e:
        print(f"Error creating point cloud: {e}")
        sys.exit(1)

    # Generate meshes for each weight (though weight is not used in this approach)
    success_count = 0
    for w in args.weight:
        out = f'{args.result}/tsdf_mesh_w{w:.1f}.ply'
        
        # Check if mesh already exists
        if os.path.exists(out) and os.path.getsize(out) > 1000:
            print(f"Mesh file {out} already exists and is valid, skipping")
            success_count += 1
            continue
        
        if create_mesh_from_pointcloud(pcd, out):
            success_count += 1
        else:
            print(f"Failed to create mesh for weight {w}")
            
            # Create empty PLY file as fallback
            try:
                with open(out, 'w') as f:
                    f.write("ply\n")
                    f.write("format ascii 1.0\n")
                    f.write("comment Mesh creation failed\n")
                    f.write("element vertex 0\n")
                    f.write("element face 0\n")
                    f.write("end_header\n")
                print(f"Created empty PLY file: {out}")
            except Exception as e:
                print(f"Error creating empty PLY file: {e}")

    if success_count > 0:
        print(f"Successfully created {success_count}/{len(args.weight)} meshes")
        sys.exit(0)
    else:
        print("Failed to create any valid meshes")
        sys.exit(1)
