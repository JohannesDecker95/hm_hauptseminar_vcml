import os
import ast
import numpy as np
import open3d as o3d
from collections import defaultdict
from glob import glob
from shutil import copyfile


out = 'outputs/TUM_RGBD'
os.makedirs(f'{out}/meshes', exist_ok=True)
seqs = sorted(glob('data/TUM_RGBD/rgbd_dataset_*')) # sorted(glob('data/TUM_RGBD/rgbd_dataset_freiburg1_desk'))
metrics = defaultdict(float)

successful_runs = 0

for seq in seqs:
    name = os.path.basename(seq)
    os.makedirs(f'{out}/{name}', exist_ok=True)
    print(name, out)

    # run HI-SLAM2
    cmd = f'CUDA_LAUNCH_BLOCKING=1 python demo.py --imagedir {seq}/colors --gtdepthdir {seq}/depths '
    cmd += f'--config config/owndata_config.yaml --calib calib/tum_rgbd.txt --output {out}/{name}'

    # if not os.path.exists(f'{out}/{name}/traj_full.txt'):
    #     print(f"Running HI-SLAM2 for {name}...")
    #     with open(f'{out}/{name}/log.txt', 'w') as log_file:
    #         try:
    #             # Use os.system instead of subprocess to inherit environment properly
    #             result = os.system(f'{cmd} > {out}/{name}/log.txt 2>&1')
    #             if result != 0:
    #                 print(f"Warning: HI-SLAM2 failed for {name} with return code {result}")
    #                 continue
    #         except Exception as e:
    #             print(f"Error running HI-SLAM2 for {name}: {e}")
    #             continue

    if not os.path.exists(f'{out}/{name}/traj_full.txt'):
        print(f"Running HI-SLAM2 command: {cmd}")
        exit_code = os.system(f'{cmd} 2>&1 | tee {out}/{name}/log.txt')
        if exit_code != 0:
            print(f"Warning: HI-SLAM2 processing may have failed (exit code: {exit_code})")
        else:
            print(f"HI-SLAM2 processing completed successfully")
    else:
        print(f"HI-SLAM2 processing already completed for {name}")


    # Check if HI-SLAM2 completed successfully
    if not os.path.exists(f'{out}/{name}/traj_full.txt'):
        print(f"HI-SLAM2 did not produce trajectory file for {name}, skipping evaluation")
        continue

    # Use groundtruth.txt if available, otherwise skip evaluation
    gt_traj_file = f'{seq}/groundtruth.txt'
    if not os.path.exists(gt_traj_file):
        print(f"No groundtruth trajectory found for {name}, skipping ATE evaluation")
        # Continue with other evaluations if possible
        try:
            # eval render (if available)
            if os.path.exists(f'{out}/{name}/psnr/after_opt/final_result.json'):
                psnr = ast.literal_eval(open(f'{out}/{name}/psnr/after_opt/final_result.json').read())
                print(f"- psnr : {psnr['mean_psnr']:.3f}, ssim: {psnr['mean_ssim']:.3f}, lpips: {psnr['mean_lpips']:.3f}")
                metrics['PSNR'] += psnr['mean_psnr']
                metrics['SSIM'] += psnr['mean_ssim']
                metrics['LPIPS'] += psnr['mean_lpips']
                successful_runs += 1
        except Exception as e:
            print(f"Error evaluating render for {name}: {e}")
        continue

    # eval ate
    if not os.path.exists(f'{out}/{name}/ape.txt') or len(open(f'{out}/{name}/ape.txt').readlines()) < 10:
        evo_cmd = f'evo_ape tum {gt_traj_file} {out}/{name}/traj_full.txt -vas --save_results {out}/{name}/evo.zip --no_warnings > {out}/{name}/ape.txt 2> {out}/{name}/evo_error.log'
        ret = os.system(evo_cmd)
        if ret != 0:
            print(f"[Error] evo_ape failed for {name}, check {out}/{name}/evo_error.log")
            continue

        os.system(f'unzip -q {out}/{name}/evo.zip -d {out}/{name}/evo')
    
    try:
        ATE = float([i for i in open(f'{out}/{name}/ape.txt').readlines() if 'rmse' in i][0][-10:-1]) * 100
        metrics['ATE full'] += ATE
        print(f'- full ATE: {ATE:.4f}')
    except (IndexError, ValueError) as e:
        print(f"Error parsing ATE for {name}: {e}")
        continue

    # eval render
    try:
        if os.path.exists(f'{out}/{name}/psnr/after_opt/final_result.json'):
            psnr = ast.literal_eval(open(f'{out}/{name}/psnr/after_opt/final_result.json').read())
            print(f"- psnr : {psnr['mean_psnr']:.3f}, ssim: {psnr['mean_ssim']:.3f}, lpips: {psnr['mean_lpips']:.3f}")
            metrics['PSNR'] += psnr['mean_psnr']
            metrics['SSIM'] += psnr['mean_ssim']
            metrics['LPIPS'] += psnr['mean_lpips']
        else:
            print("- Render evaluation file not found, skipping PSNR/SSIM/LPIPS")
    except Exception as e:
        print(f"Error evaluating render for {name}: {e}")

    # run tsdf fusion with memory-aware parameters
    w = 2
    weight = f'w{w:.1f}'
    if not os.path.exists(f'{out}/{name}/tsdf_mesh_{weight}.ply'):
        try:
            # Check number of renders to determine parameters
            render_count = 0
            if os.path.exists(f'{out}/{name}/renders/depth_after_opt'):
                render_count = len(glob(f'{out}/{name}/renders/depth_after_opt/*.png'))
            
            # Use larger voxel size and smaller block count for datasets with many frames
            # Special handling for known problematic datasets
            if name in ['rgbd_dataset_freiburg1_room', 'rgbd_dataset_freiburg3_long_office_household']:
                voxel_size = 0.08   # Very large voxel size for memory efficiency
                block_count = 10000  # Very small block count
                print(f"Using ultra-conservative parameters for problematic dataset {name}")
            elif render_count > 400:
                voxel_size = 0.008  # Larger voxel size for memory efficiency
                block_count = 30000  # Smaller block count
                print(f"Using memory-optimized parameters for {name} ({render_count} frames)")
            else:
                voxel_size = 0.006  # Standard voxel size
                block_count = 50000  # Standard block count
            
            cmd = f'python tsdf_integrate.py --result {out}/{name} --voxel_size {voxel_size} --weight {w} --block_count {block_count}'
            result = os.system(f'{cmd} > /dev/null 2>&1')
            
            if result == 0 and os.path.exists(f'{out}/{name}/tsdf_mesh_{weight}.ply'):
                ply = o3d.io.read_triangle_mesh(f'{out}/{name}/tsdf_mesh_{weight}.ply')
                if os.path.exists(f'{out}/{name}/evo/alignment_transformation_sim3.npy'):
                    ply = ply.transform(np.load(f'{out}/{name}/evo/alignment_transformation_sim3.npy'))
                    o3d.io.write_triangle_mesh(f'{out}/{name}/tsdf_mesh_{weight}_aligned.ply', ply)
                    copyfile(f'{out}/{name}/tsdf_mesh_{weight}_aligned.ply', f'{out}/meshes/{name}.ply')
                else:
                    print(f"Warning: No alignment transformation found for {name}")
                    copyfile(f'{out}/{name}/tsdf_mesh_{weight}.ply', f'{out}/meshes/{name}.ply')
            else:
                print(f"Warning: TSDF mesh generation failed for {name}")
        except Exception as e:
            print(f"Error in TSDF fusion for {name}: {e}")
    
    # 3D reconstruction evaluation
    if os.path.exists(f'data/TUM_RGBD/gt_mesh_culled/{name}.ply'):
        if not os.path.exists(f'{out}/{name}/eval_recon_{weight}.txt'):
            mesh_file = f'{out}/{name}/tsdf_mesh_{weight}_aligned.ply' if os.path.exists(f'{out}/{name}/tsdf_mesh_{weight}_aligned.ply') else f'{out}/{name}/tsdf_mesh_{weight}.ply'
            if os.path.exists(mesh_file):
                eval_cmd = f'python scripts/eval_recon.py {mesh_file} data/TUM_RGBD/gt_mesh_culled/{name}.ply --eval_3d --save {out}/{name}/eval_recon_{weight}.txt'
                result = os.system(f'{eval_cmd} > {out}/{name}/eval_recon_log.txt 2>&1')
                if result != 0:
                    print(f"Warning: 3D reconstruction evaluation had issues for {name}, check {out}/{name}/eval_recon_log.txt")
        
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
        print("- 3D reconstruction evaluation skipped (no ground truth mesh)")
    
    successful_runs += 1
    print(f"✓ Completed processing for {name}\n")

print(f"\n=== Final Results (averaged over {successful_runs} successful runs) ===")
if successful_runs > 0:
    for r in metrics:
        print(f'{r}: \t {metrics[r]/successful_runs:.4f}')
else:
    print("No successful runs to report")
