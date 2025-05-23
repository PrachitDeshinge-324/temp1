echo "Extracting database information..."
python main.py \
  --video "../Person_New/input/3c.mp4" \
  --output "results3/3c.mp4" \
  --results_dir "results3" \
  --save_bbox_info \
  --merge_ids \
  --use_transreid \
  --save_video 
