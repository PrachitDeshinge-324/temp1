echo "Extracting database information..."
python main.py \
  --video "../Person_New/input/3c.mp4" \
  --results_dir "results3" \
  --output_video "results3/3c.mp4" \
  --save_bbox_info \
  --merge_ids \
  --end_frame 250 \
  --start_frame 50 \
  --headless
