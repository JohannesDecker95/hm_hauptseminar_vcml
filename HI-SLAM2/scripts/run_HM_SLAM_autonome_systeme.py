import os
import ast
import numpy as np
import open3d as o3d
from collections import defaultdict
from glob import glob
from shutil import copyfile


out = 'outputs/HM_SLAM_autonome_systeme'
os.makedirs(f'{out}/meshes', exist_ok=True)
seqs = sorted(glob('data/HM_SLAM_autonome_systeme/06*')) + sorted(glob('data/HM_SLAM_autonome_systeme/26*'))
metrics = defaultdict(float)
for seq in seqs:
    name = os.path.basename(seq)
    os.makedirs(f'{out}/{name}', exist_ok=True)
    print(name, out)

    # run HI-SLAM2
    cmd = f'python demo.py --imagedir {seq}/colors --gtdepthdir {seq}/depths '
    cmd += f'--config config/owndata_config.yaml --calib calib/hm_slam.txt --output {out}/{name}'
    
    if not os.path.exists(f'{out}/{name}/traj_full.txt'):
        print(f"Running HI-SLAM2 command: {cmd}")
        exit_code = os.system(f'{cmd} 2>&1 | tee {out}/{name}/log.txt')
        if exit_code != 0:
            print(f"Warning: HI-SLAM2 processing may have failed (exit code: {exit_code})")
        else:
            print(f"HI-SLAM2 processing completed successfully")
    else:
        print(f"HI-SLAM2 processing already completed for {name}")

    # eval ate
    if not os.path.exists(f'{out}/{name}/ape.txt') or len(open(f'{out}/{name}/ape.txt').readlines()) < 10:
        os.system(f'evo_ape tum {seq}/traj_tum.txt {out}/{name}/traj_full.txt -vas --save_results {out}/{name}/evo.zip --no_warnings > {out}/{name}/ape.txt')
        os.system(f'unzip -q {out}/{name}/evo.zip -d {out}/{name}/evo')
    
    # Parse ATE with error handling
    try:
        ape_lines = open(f'{out}/{name}/ape.txt').readlines()
        rmse_lines = [i for i in ape_lines if 'rmse' in i]
        if rmse_lines:
            ATE = float(rmse_lines[0][-10:-1]) * 100
            metrics['ATE full'] += ATE
            print(f'- full ATE: {ATE:.4f}')
        else:
            print(f'- full ATE: Error - no RMSE found in ape.txt')
    except (IndexError, ValueError, FileNotFoundError) as e:
        print(f'- full ATE: Error parsing ape.txt - {e}')

    # eval render
    try:
        psnr = ast.literal_eval(open(f'{out}/{name}/psnr/after_opt/final_result.json').read())
        print(f"- psnr : {psnr['mean_psnr']:.3f}, ssim: {psnr['mean_ssim']:.3f}, lpips: {psnr['mean_lpips']:.3f}")
        metrics['PSNR'] += psnr['mean_psnr']
        metrics['SSIM'] += psnr['mean_ssim']
        metrics['LPIPS'] += psnr['mean_lpips']
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"- render metrics: Error reading final_result.json - {e}")

    # run tsdf fusion - check if we have the required files first
    w = 2
    weight = f'w{w:.1f}'
    mesh_file = f'{out}/{name}/tsdf_mesh_{weight}.ply'
    aligned_mesh_file = f'{out}/{name}/tsdf_mesh_{weight}_aligned.ply'
    
    # Check if the HI-SLAM2 processing actually generated the required files
    required_files = [
        f'{out}/{name}/renders/depth_after_opt',
        f'{out}/{name}/renders/image_after_opt',
        f'{out}/{name}/intrinsics.npy',
        f'{out}/{name}/traj_full.txt'
    ]
    
    files_exist = all(os.path.exists(f) for f in required_files)
    
    if files_exist:
        # Check if we have actual depth and color files
        depth_files = glob(f'{out}/{name}/renders/depth_after_opt/*.png')
        color_files = glob(f'{out}/{name}/renders/image_after_opt/*.jpg')
        
        if len(depth_files) > 0 and len(color_files) > 0:
            print(f"Found {len(depth_files)} depth and {len(color_files)} color files for TSDF integration")
            
            if not os.path.exists(mesh_file):
                print(f"Running TSDF-free mesh generation using HM_SLAM safe version...")
                tsdf_exit = os.system(f'python tsdf_integrate_hm_slam_safe.py --result {out}/{name} --voxel_size 0.006 --weight {w}')
                
                # Check the actual exit code from the mesh generation
                # tsdf_exit = 0
                
                if tsdf_exit == 0 and os.path.exists(mesh_file) and os.path.getsize(mesh_file) > 1000:
                    print(f"TSDF mesh generation successful: {mesh_file}")
                else:
                    print(f"TSDF mesh generation failed or produced empty mesh (exit code: {tsdf_exit})")
            else:
                print(f"TSDF mesh already exists: {mesh_file}")
            
            # Try to create aligned mesh (regardless of whether mesh was just generated or already existed)
            if os.path.exists(mesh_file) and os.path.getsize(mesh_file) > 1000:
                try:
                    ply = o3d.io.read_triangle_mesh(mesh_file)
                    if len(ply.vertices) > 0:
                        alignment_file = f'{out}/{name}/evo/alignment_transformation_sim3.npy'
                        if os.path.exists(alignment_file):
                            ply = ply.transform(np.load(alignment_file))
                            o3d.io.write_triangle_mesh(aligned_mesh_file, ply)
                            copyfile(aligned_mesh_file, f'{out}/meshes/{name}.ply')
                            print(f"Aligned mesh created: {aligned_mesh_file}")
                        else:
                            print("No alignment transformation found, using unaligned mesh")
                            copyfile(mesh_file, f'{out}/meshes/{name}.ply')
                    else:
                        print("Warning: TSDF mesh has no vertices")
                except Exception as e:
                    print(f"Error processing TSDF mesh: {e}")
        else:
            print(f"Missing rendered files - depth: {len(depth_files)}, color: {len(color_files)}")
            print(f"HI-SLAM2 processing may not have completed successfully")
    else:
        print(f"Required files missing for TSDF generation:")
        for f in required_files:
            status = "✓" if os.path.exists(f) else "✗"
            print(f"  {status} {f}")
    
    # eval 3d reconstruction
    gt_mesh_path = f'data/HM_SLAM_autonome_systeme/gt_mesh_culled/{name}.ply'
    if os.path.exists(gt_mesh_path):
        eval_mesh_file = aligned_mesh_file if os.path.exists(aligned_mesh_file) else mesh_file
        if os.path.exists(eval_mesh_file) and os.path.getsize(eval_mesh_file) > 1000:
            if not os.path.exists(f'{out}/{name}/eval_recon_{weight}.txt'):
                print(f"Running 3D reconstruction evaluation...")
                eval_cmd = f'python scripts/eval_recon.py {eval_mesh_file} {gt_mesh_path} --eval_3d --save {out}/{name}/eval_recon_{weight}.txt'
                eval_result = os.system(f'{eval_cmd} > {out}/{name}/eval_recon_log.txt 2>&1')
                if eval_result != 0:
                    print(f"Warning: 3D reconstruction evaluation had issues, check {out}/{name}/eval_recon_log.txt")
            
            if os.path.exists(f'{out}/{name}/eval_recon_{weight}.txt'):
                try:
                    result = ast.literal_eval(open(f'{out}/{name}/eval_recon_{weight}.txt').read())
                    metrics['accu'] += result['mean precision']
                    metrics['comp'] += result['mean recall']
                    metrics['compr'] += result['recall']
                    print(f"- acc: {result['mean precision']:.3f}, comp: {result['mean recall']:.3f}, comp rat: {result['recall']:.3f}")
                except Exception as e:
                    print(f"Error evaluating 3D reconstruction for {name}: {e}")
            else:
                print("- 3D reconstruction evaluation failed or incomplete")
        else:
            print("- 3D reconstruction evaluation skipped (no valid TSDF mesh)")
    else:
        print("- 3D reconstruction evaluation skipped (no ground truth mesh)")
    
    # Fallback: create placeholder files if TSDF generation wasn't possible
    if not os.path.exists(mesh_file):
        print(f"Creating placeholder mesh files...")
        for placeholder_file in [mesh_file, aligned_mesh_file, f'{out}/meshes/{name}.ply']:
            if not os.path.exists(placeholder_file):
                os.makedirs(os.path.dirname(placeholder_file), exist_ok=True)
                try:
                    with open(placeholder_file, 'w') as f:
                        f.write("ply\nformat ascii 1.0\ncomment TSDF generation incomplete\nelement vertex 0\nelement face 0\nend_header\n")
                except Exception:
                    pass
    
    # Report final status
    if os.path.exists(f'{out}/{name}/3dgs_final.ply'):
        print("- 3D reconstruction: 3D Gaussian Splatting completed successfully")
        print(f"- 3DGS mesh: {out}/{name}/3dgs_final.ply")
    else:
        print("- Warning: 3DGS mesh not found")
        
    if os.path.exists(mesh_file) and os.path.getsize(mesh_file) > 1000:  # Check if it's not just a placeholder
        print(f"- TSDF mesh: {mesh_file}")
    else:
        print("- TSDF mesh: generation incomplete or failed")
        
    print()

print(f"\n=== Final Results (averaged over {len(seqs)} sequences) ===")
for r in metrics:
    print(f'{r}: \t {metrics[r]/len(seqs):.4f}')
