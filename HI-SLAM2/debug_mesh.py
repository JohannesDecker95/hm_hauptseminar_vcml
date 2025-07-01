import open3d as o3d
import numpy as np

# Load reconstructed mesh
print("Loading reconstructed mesh...")
recon_mesh = o3d.io.read_triangle_mesh('outputs/TUM_RGBD/rgbd_dataset_freiburg1_desk/tsdf_mesh_w2.0_aligned.ply')
print(f'Reconstructed mesh: {len(recon_mesh.vertices)} vertices, {len(recon_mesh.triangles)} triangles')

# Load ground truth mesh
print("Loading ground truth mesh...")
gt_mesh = o3d.io.read_triangle_mesh('data/TUM_RGBD/gt_mesh_culled/rgbd_dataset_freiburg1_desk.ply')
print(f'Ground truth mesh: {len(gt_mesh.vertices)} vertices, {len(gt_mesh.triangles)} triangles')

# Check if meshes are empty
if len(recon_mesh.vertices) == 0:
    print("ERROR: Reconstructed mesh has no vertices!")
    exit(1)

if len(gt_mesh.vertices) == 0:
    print("ERROR: Ground truth mesh has no vertices!")
    exit(1)

# Check bounding boxes
recon_bbox = recon_mesh.get_axis_aligned_bounding_box()
gt_bbox = gt_mesh.get_axis_aligned_bounding_box()

print(f'Reconstructed mesh bounds: min={np.array(recon_bbox.min_bound)}, max={np.array(recon_bbox.max_bound)}')
print(f'Ground truth mesh bounds: min={np.array(gt_bbox.min_bound)}, max={np.array(gt_bbox.max_bound)}')

# Check scales
recon_extent = np.array(recon_bbox.max_bound) - np.array(recon_bbox.min_bound)
gt_extent = np.array(gt_bbox.max_bound) - np.array(gt_bbox.min_bound)

print(f'Reconstructed mesh extent: {recon_extent}')
print(f'Ground truth mesh extent: {gt_extent}')

# Check if extents are reasonable
if np.any(recon_extent < 1e-6):
    print("WARNING: Reconstructed mesh has very small extent!")
if np.any(gt_extent < 1e-6):
    print("WARNING: Ground truth mesh has very small extent!")

# Check distance between centers
recon_center = np.array(recon_bbox.get_center())
gt_center = np.array(gt_bbox.get_center())
distance_between_centers = np.linalg.norm(recon_center - gt_center)

print(f'Reconstructed mesh center: {recon_center}')
print(f'Ground truth mesh center: {gt_center}')
print(f'Distance between centers: {distance_between_centers}')

# Check scale ratios
scale_ratios = recon_extent / gt_extent
print(f'Scale ratios (recon/gt): {scale_ratios}')

if np.any(scale_ratios > 100) or np.any(scale_ratios < 0.01):
    print("WARNING: Large scale difference detected!")

print("Mesh analysis complete.")
