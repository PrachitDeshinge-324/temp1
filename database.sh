echo "Extracting database information..."
python main.py \
  --video "../Person_New/input/3c.mp4" \
  --results_dir "results1" \
  --output_video "results1/3c.mp4" \
  --use_deepsort \
  --save_bbox_info \
  --merge_ids \
  --end_frame 350 \
  --start_frame 50
