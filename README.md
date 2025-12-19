# TrackNet V4 PyTorch

A PyTorch implementation of **TrackNet V4: Enhancing Fast Sports Object Tracking with Motion Attention Maps** for real-time tracking of small, fast-moving objects in sports videos.

## Overview

TrackNet V4 enhances sports object tracking by incorporating motion attention maps that focus on temporal changes between consecutive frames. The model excels at tracking small, fast-moving objects like tennis balls and ping-pong balls in challenging scenarios with occlusion and motion blur.

**Key Features:**
- Motion-aware tracking with attention mechanisms
- Real-time video processing capabilities  
- Robust handling of occlusion and motion blur
- End-to-end training pipeline

## Requirements

- Python ≥ 3.10
- PyTorch ≥ 1.9.0
- CUDA (recommended for training)

## Installation

```bash
git clone https://github.com/AnInsomniacy/tracknet-v4-pytorch.git
cd tracknet-v4-pytorch
pip install -r requirements.txt
```

## Configuration

All parameters are configured in `config.yaml`. Edit this file to customize preprocessing, training, and testing settings.

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

### TensorBoard
```bash
tensorboard --logdir outputs/
```

### Inference
```bash
PYTHONPATH=. python predict/video_predict.py
PYTHONPATH=. python predict/single_frame_predict.py
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
│   ├── tracknet_v4.py          # Main TrackNet V4 architecture
│   ├── tracknet_v2.py          # Legacy TrackNet V2
│   └── loss.py                 # Weighted Binary Cross Entropy loss
├── preprocessing/
│   ├── tracknet_dataset.py     # PyTorch dataset loader
│   └── data_visualizer.py      # Data visualization tools
├── predict/
│   ├── single_frame_predict.py # Single frame inference
│   └── video_predict.py        # Video batch processing
├── config.yaml                 # Configuration file
├── preprocess.py               # Dataset preprocessing
├── train.py                    # Training script
├── test.py                     # Model evaluation
└── requirements.txt            # Dependencies
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