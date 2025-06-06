#!/bin/bash
# Set environment variable for PyTorch MPS fallback
export PYTORCH_ENABLE_MPS_FALLBACK=1
# Delete all __pycache__ and .cache directories
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type d -name ".cache" -exec rm -rf {} +

# Run the main script with all arguments
python main.py \
    --input '../Person_New/input/3c.mp4' \
    --output_dir 'output' \
    --weights_dir '../Fresh/weights' \
    --transreid_weights '../Fresh/weights/transreid_vitbase.pth' \
    --display \
    --gait_analysis \
    --opengait_config "OpenGait/configs/deepgaitv2/DeepGaitV2_sustech1k.yaml"\
    --opengait_weights 'DeepGaitV2_30_DA-50000.pt' \
    --build_gallery

find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type d -name ".cache" -exec rm -rf {} +