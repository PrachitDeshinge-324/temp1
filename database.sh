echo "Extracting database information..."
python main.py \
    --output_video "results/3c.mp4" \
    --use_deepsort \
    --save_bbox_info \
    --merge_ids \
    --start_frame 0 \
    --end_frame 1000 \