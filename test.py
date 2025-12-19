"""TrackNet Testing Script

Usage:
    python test.py --config config.yaml
"""

import sys
import time
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model.tracknet_v4 import TrackNet
from preprocessing.tracknet_dataset import FrameHeatmapDataset

GROUND_TRUTH_THRESHOLD = 0.1


def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)['test']


class TrackNetTester:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = self._setup_device()
        self._setup_dirs()
        self._load_model()
        self.frame_predictions = {}
        self.results = {'tp': 0, 'tn': 0, 'fp1': 0, 'fp2': 0, 'fn': 0, 'total_frames': 0, 'detected_frames': 0}

    def _setup_device(self):
        if self.cfg['device'] == 'auto':
            if torch.backends.mps.is_available():
                return torch.device('mps')
            elif torch.cuda.is_available():
                return torch.device('cuda')
            return torch.device('cpu')
        return torch.device(self.cfg['device'])

    def _setup_dirs(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = Path(self.cfg['out']) / f"test_{timestamp}"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.save_dir / "tensorboard"))

    def _load_model(self):
        self.model = TrackNet().to(self.device)
        checkpoint = torch.load(self.cfg['model'], map_location=self.device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def _setup_data(self):
        self.test_dataset = FrameHeatmapDataset(self.cfg['data'])
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=self.cfg['batch'], shuffle=False,
            num_workers=0, pin_memory=self.device.type == 'cuda'
        )

    def _extract_coordinates(self, heatmap):
        if heatmap.max() < self.cfg['threshold']:
            return None
        max_pos = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        return (max_pos[1], max_pos[0])

    def _extract_ground_truth_coordinates(self, target_heatmap):
        if target_heatmap.max() < GROUND_TRUTH_THRESHOLD:
            return None
        max_pos = np.unravel_index(np.argmax(target_heatmap), target_heatmap.shape)
        return (max_pos[1], max_pos[0])

    def _calculate_distance(self, pred_coord, gt_coord):
        if pred_coord is None or gt_coord is None:
            return float('inf')
        return np.sqrt((pred_coord[0] - gt_coord[0]) ** 2 + (pred_coord[1] - gt_coord[1]) ** 2)

    def _classify_prediction(self, pred_coord, gt_coord):
        has_pred = pred_coord is not None
        has_gt = gt_coord is not None
        if not has_pred and not has_gt:
            return 'tn'
        elif not has_pred and has_gt:
            return 'fn'
        elif has_pred and not has_gt:
            return 'fp2'
        else:
            distance = self._calculate_distance(pred_coord, gt_coord)
            return 'tp' if distance <= self.cfg['tolerance'] else 'fp1'

    def _collect_predictions(self, outputs, targets, batch_start_idx):
        batch_size = outputs.shape[0]
        for b in range(batch_size):
            item_info = self.test_dataset.get_info(batch_start_idx + b)
            base_frame_idx = item_info['idx']
            match_name = item_info['match']
            frame_name = item_info['frame']
            for f in range(3):
                pred_heatmap = outputs[b, f].cpu().numpy()
                gt_heatmap = targets[b, f].cpu().numpy()
                pred_coord = self._extract_coordinates(pred_heatmap)
                gt_coord = self._extract_ground_truth_coordinates(gt_heatmap)
                frame_idx = base_frame_idx + f
                frame_key = f"{match_name}_{frame_name}_{frame_idx}"
                if frame_key not in self.frame_predictions:
                    self.frame_predictions[frame_key] = []
                self.frame_predictions[frame_key].append({
                    'pred': pred_coord,
                    'gt': gt_coord,
                    'is_center': f == 1,
                    'distance': self._calculate_distance(pred_coord, gt_coord)
                })

    def _process_center_frame_predictions(self):
        for frame_key, predictions in self.frame_predictions.items():
            center_pred = None
            for pred in predictions:
                if pred['is_center']:
                    center_pred = pred
                    break
            if center_pred:
                classification = self._classify_prediction(center_pred['pred'], center_pred['gt'])
                self.results[classification] += 1
                self.results['total_frames'] += 1
                if center_pred['pred'] is not None:
                    self.results['detected_frames'] += 1

    def _calculate_metrics(self):
        tp, tn, fp1, fp2, fn = self.results['tp'], self.results['tn'], self.results['fp1'], self.results['fp2'], self.results['fn']
        total_fp = fp1 + fp2
        total_predictions = tp + total_fp
        total_positives = tp + fn
        total_samples = tp + tn + total_fp + fn
        accuracy = (tp + tn) / total_samples if total_samples > 0 else 0
        precision = tp / total_predictions if total_predictions > 0 else 0
        recall = tp / total_positives if total_positives > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        detection_rate = self.results['detected_frames'] / self.results['total_frames'] if self.results['total_frames'] > 0 else 0
        return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1_score': f1_score, 'detection_rate': detection_rate}

    def _log_metrics(self, metrics):
        self.writer.add_scalar('metrics/accuracy', metrics['accuracy'], 0)
        self.writer.add_scalar('metrics/precision', metrics['precision'], 0)
        self.writer.add_scalar('metrics/recall', metrics['recall'], 0)
        self.writer.add_scalar('metrics/f1_score', metrics['f1_score'], 0)
        self.writer.add_scalar('metrics/detection_rate', metrics['detection_rate'], 0)
        self.writer.add_scalar('confusion/tp', self.results['tp'], 0)
        self.writer.add_scalar('confusion/tn', self.results['tn'], 0)
        self.writer.add_scalar('confusion/fp1', self.results['fp1'], 0)
        self.writer.add_scalar('confusion/fp2', self.results['fp2'], 0)
        self.writer.add_scalar('confusion/fn', self.results['fn'], 0)

    def run_test(self):
        self._setup_data()
        
        print(f"TensorBoard: tensorboard --logdir {self.save_dir / 'tensorboard'}")
        
        start_time = time.time()
        batch_start_idx = 0
        with torch.no_grad():
            for inputs, targets in tqdm(self.test_loader, desc="Testing"):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs = self.model(inputs)
                self._collect_predictions(outputs, targets, batch_start_idx)
                batch_start_idx += inputs.shape[0]
        self._process_center_frame_predictions()
        test_time = time.time() - start_time
        metrics = self._calculate_metrics()
        self._log_metrics(metrics)
        self.writer.add_scalar('test/time', test_time, 0)
        self.writer.add_scalar('test/fps', self.results['total_frames'] / test_time, 0)
        self.writer.close()
        return metrics


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    if not Path(config['model']).exists():
        sys.exit(1)
    if not Path(config['data']).exists():
        sys.exit(1)
    tester = TrackNetTester(config)
    tester.run_test()
