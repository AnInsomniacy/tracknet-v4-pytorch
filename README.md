# TrackNet Series PyTorch

A PyTorch implementation of the **TrackNet Series** for real-time tracking of small, fast-moving objects in sports videos.

## Overview

This repository implements multiple versions of TrackNet for sports object tracking:

- ✅ **TrackNet V2** - U-Net baseline with VGG-style encoder
- ✅ **TrackNet V4** - Motion attention enhanced tracking
- 🚧 **TrackNet V3** - Coming soon

**Key Features:**
- Multi-GPU DDP training support
- Real-time video processing capabilities  
- Robust handling of occlusion and motion blur
- End-to-end training pipeline

## Requirements

- Python ≥ 3.10
- PyTorch ≥ 1.9.0
- CUDA (recommended for training)

## Configuration

All parameters are configured in `config.yaml`. Edit this file to customize preprocessing, training, testing, and prediction settings.

## Usage

### Data Preprocessing
```bash
python preprocess.py --config config.yaml
```

### Training
```bash
python train.py --config config.yaml
```

### Testing
```bash
python test.py --config config.yaml
```

### Prediction
```bash
python predict.py --config config.yaml
```

### Predict with visualization
```bash
# Video prediction
PYTHONPATH=. python predict/video_predict.py

# Single frame prediction  
PYTHONPATH=. python predict/single_frame_predict.py

# Stream video  prediction without  visualize

PYTHONPATH=. python run predict/streem_video_predict.py --model_path checkpoints/best_model.pth  --video_path demo.mp4 --output_dir ./predict_video

# Stream video  prediction with  visualize

PYTHONPATH=. python run predict/streem_video_predict.py --model_path checkpoints/best_model.pth  --video_path demo.mp4 --output_dir ./predict_video --visualize

# Stream video  prediction save only predict.csv
PYTHONPATH=. python run predict/streem_video_predict.py --model_path checkpoints/best_model.pth  --video_path demo.mp4 --output_dir ./predict_video --only_csv
```

### TensorBoard
```bash
tensorboard --logdir outputs/
```

## Model Architecture

TrackNet V4 introduces motion attention to enhance tracking performance:

- **Input:** 3 consecutive RGB frames (9 channels, 288×512)
- **Motion Prompt Layer:** Extracts motion attention from frame differences  
- **Encoder-Decoder:** VGG-style architecture with skip connections
- **Output:** Object probability heatmaps (3 channels, 288×512)

The motion attention mechanism focuses on regions with significant temporal changes, improving detection of fast-moving objects.

## Data Format

**Input Structure:**
```
dataset/
├── inputs/          # RGB frames (288×512)
└── heatmaps/        # Ground truth heatmaps (288×512)
```

- Input: 3 consecutive frames concatenated into 9-channel tensors
- Heatmaps: Gaussian distributions centered on object locations

## Project Structure

```
tracknet-v4-pytorch/
├── model/
│   ├── tracknet_v4.py      # TrackNet V4 with motion attention
│   ├── tracknet_v2.py      # TrackNet V2 baseline
│   ├── tracknet_exp.py     # Experimental model with CBAM
│   └── loss.py             # Weighted Binary Cross Entropy loss
├── preprocessing/
│   ├── tracknet_dataset.py # PyTorch dataset loader
│   └── data_visualizer.py  # Data visualization tools
├── config.yaml             # Configuration file
├── preprocess.py           # Dataset preprocessing
├── train.py                # Training script
├── test.py                 # Model evaluation
├── predict.py              # Video inference
└── requirements.txt        # Dependencies
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{raj2024tracknetv4,
    title={TrackNetV4: Enhancing Fast Sports Object Tracking with Motion Attention Maps},
    author={Raj, Arjun and Wang, Lei and Gedeon, Tom},
    journal={arXiv preprint arXiv:2409.14543},
    year={2024}
}
```

## License

This project is available for research and educational purposes.
