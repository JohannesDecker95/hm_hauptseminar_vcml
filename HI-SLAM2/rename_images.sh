for file in /workspace/HI-SLAM2/data/HM_SLAM_autonome_systeme/060525_SLAM_1_2_video/results/*; do
  [ -f "$file" ] && mv "$file" "/workspace/HI-SLAM2/data/HM_SLAM_autonome_systeme/060525_SLAM_1_2_video/results/frame_$(basename "$file")"
done

# 060525_SLAM_1_2_video

# for file in /workspace/HI-SLAM2/data/HM_SLAM_autonome_systeme/260525_SLAM_LONG_VARIOUS_SPEED_2_video/depth/*; do
#   if [ -f "$file" ]; then
#     filename=$(basename "$file")
#     newname="${filename%%-*}.png"
#     mv "$file" "/workspace/HI-SLAM2/data/HM_SLAM_autonome_systeme/260525_SLAM_LONG_VARIOUS_SPEED_2_video/depth/$newname"
#   fi
# done