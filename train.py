"""TrackNet Training Script

Usage:
    python train.py --config config.yaml
"""

import signal
import sys
import time
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import torch
import yaml
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model.loss import WeightedBinaryCrossEntropy
from model.tracknet_v2 import TrackNet
from preprocessing.tracknet_dataset import FrameHeatmapDataset


def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)['train']


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.start_epoch = 0
        self.interrupted = False
        self.best_loss = float('inf')
        self.device = self._get_device()
        self.step = 0
        self.checkpoint = None
        self._setup_dirs()
        signal.signal(signal.SIGINT, self._interrupt)
        signal.signal(signal.SIGTERM, self._interrupt)

    def _get_device(self):
        if self.cfg['device'] == 'auto':
            if torch.backends.mps.is_available():
                return torch.device('mps')
            elif torch.cuda.is_available():
                return torch.device('cuda')
            return torch.device('cpu')
        return torch.device(self.cfg['device'])

    def _setup_dirs(self):
        resume_dir = self.cfg.get('resume')
        if resume_dir:
            self.save_dir = Path(resume_dir)
            if not self.save_dir.exists():
                raise FileNotFoundError(f"Resume directory not found: {self.save_dir}")
            checkpoint_dir = self.save_dir / "checkpoints"
            checkpoint_files = sorted(checkpoint_dir.glob("checkpoint_*.pth"))
            if not checkpoint_files:
                checkpoint_files = sorted(checkpoint_dir.glob("emergency_*.pth"))
            if not checkpoint_files:
                raise FileNotFoundError(f"No checkpoint found in: {checkpoint_dir}")
            latest_checkpoint = checkpoint_files[-1]
            self.checkpoint = torch.load(latest_checkpoint, map_location='cpu')
            self.start_epoch = self.checkpoint['epoch'] + (0 if self.checkpoint.get('is_emergency', False) else 1)
            self.step = self.checkpoint.get('step', 0)
            self.best_loss = self.checkpoint.get('best_loss', self.checkpoint.get('val_loss', float('inf')))
            self.writer = SummaryWriter(log_dir=str(self.save_dir / "tensorboard"))
            print(f"Resuming from epoch {self.start_epoch}, step {self.step}")
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.save_dir = Path(self.cfg['out']) / f"{self.cfg['name']}_{timestamp}"
            self.save_dir.mkdir(parents=True, exist_ok=True)
            (self.save_dir / "checkpoints").mkdir(exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(self.save_dir / "tensorboard"))

    def _interrupt(self, signum, frame):
        self.interrupted = True

    def _get_lr(self):
        if self.cfg['lr'] is not None:
            return self.cfg['lr']
        defaults = {'Adadelta': 1.0, 'Adam': 0.001, 'AdamW': 0.001, 'SGD': 0.01}
        return defaults[self.cfg['optimizer']]

    def _calculate_effective_lr(self):
        if self.cfg['optimizer'] == 'Adadelta':
            if not hasattr(self.optimizer, 'state') or not self.optimizer.state:
                return self._get_lr()
            effective_lrs = []
            eps = self.optimizer.param_groups[0].get('eps', 1e-6)
            for group in self.optimizer.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    state = self.optimizer.state[p]
                    if len(state) == 0:
                        continue
                    square_avg = state.get('square_avg')
                    acc_delta = state.get('acc_delta')
                    if square_avg is not None and acc_delta is not None:
                        if torch.is_tensor(square_avg) and torch.is_tensor(acc_delta):
                            rms_delta = (acc_delta + eps).sqrt().mean()
                            rms_grad = (square_avg + eps).sqrt().mean()
                            if rms_grad > eps:
                                effective_lr = self._get_lr() * rms_delta / rms_grad
                                effective_lrs.append(effective_lr.item())
            if effective_lrs:
                return max(sum(effective_lrs) / len(effective_lrs), eps)
            return self._get_lr()
        return self.optimizer.param_groups[0]['lr']

    def setup_data(self):
        dataset = FrameHeatmapDataset(self.cfg['data'])
        torch.manual_seed(self.cfg['seed'])
        train_size = int(self.cfg['split'] * len(dataset))
        train_ds, val_ds = random_split(dataset, [train_size, len(dataset) - train_size])
        self.train_loader = DataLoader(train_ds, batch_size=self.cfg['batch'], shuffle=True,
                                       num_workers=self.cfg['workers'], pin_memory=self.device.type == 'cuda')
        self.val_loader = DataLoader(val_ds, batch_size=self.cfg['batch'], shuffle=False,
                                     num_workers=self.cfg['workers'], pin_memory=self.device.type == 'cuda')
        if not self.checkpoint:
            self.writer.add_text('dataset/train_size', str(len(train_ds)))
            self.writer.add_text('dataset/val_size', str(len(val_ds)))

    def _create_optimizer(self):
        lr = self._get_lr()
        wd = self.cfg['wd']
        optimizers = {
            'Adadelta': lambda: torch.optim.Adadelta(self.model.parameters(), lr=lr, weight_decay=wd),
            'Adam': lambda: torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd),
            'AdamW': lambda: torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd),
            'SGD': lambda: torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
        }
        return optimizers[self.cfg['optimizer']]()

    def setup_model(self):
        self.model = TrackNet().to(self.device)
        self.criterion = WeightedBinaryCrossEntropy()
        self.optimizer = self._create_optimizer()
        if self.cfg['scheduler'] == "ReduceLROnPlateau":
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=self.cfg['factor'],
                                               patience=self.cfg['patience'], min_lr=self.cfg['min_lr'])
        else:
            self.scheduler = None
        if self.checkpoint:
            self.model.load_state_dict(self.checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in self.checkpoint:
                self.optimizer.load_state_dict(self.checkpoint['optimizer_state_dict'])
            if self.scheduler and 'scheduler_state_dict' in self.checkpoint:
                self.scheduler.load_state_dict(self.checkpoint['scheduler_state_dict'])

    def save_checkpoint(self, epoch, train_loss, val_loss, is_emergency=False):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'best_loss': self.best_loss,
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'is_emergency': is_emergency,
            'step': self.step,
            'timestamp': timestamp
        }
        prefix = "emergency_" if is_emergency else "checkpoint_"
        filename = f"{prefix}epoch_{epoch + 1}_{timestamp}.pth"
        filepath = self.save_dir / "checkpoints" / filename
        torch.save(checkpoint, filepath)
        if not is_emergency and val_loss < self.best_loss:
            self.best_loss = val_loss
            checkpoint['best_loss'] = self.best_loss
            torch.save(checkpoint, self.save_dir / "checkpoints" / "best_model.pth")
            return filepath, True
        return filepath, False

    def validate(self):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for inputs, targets in tqdm(self.val_loader, desc="Validating", leave=False):
                if self.interrupted:
                    break
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
        return total_loss / len(self.val_loader)

    def train(self):
        self.setup_data()
        self.setup_model()
        
        print(f"TensorBoard: tensorboard --logdir {self.save_dir / 'tensorboard'}")
        
        for epoch in range(self.start_epoch, self.cfg['epochs']):
            if self.interrupted:
                break
            start_time = time.time()
            self.model.train()
            total_loss = 0.0
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.cfg['epochs']}", leave=False)
            for batch_idx, (inputs, targets) in enumerate(pbar):
                if self.interrupted:
                    pbar.close()
                    val_loss = self.validate()
                    self.save_checkpoint(epoch, total_loss / (batch_idx + 1), val_loss, True)
                    self.writer.close()
                    return
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                batch_loss = loss.item()
                total_loss += batch_loss
                self.step += 1
                current_lr = self._calculate_effective_lr()
                self.writer.add_scalar('batch/loss', batch_loss, self.step)
                self.writer.add_scalar('batch/lr', current_lr, self.step)
                pbar.set_postfix({'loss': f'{batch_loss:.6f}'})
            pbar.close()
            train_loss = total_loss / len(self.train_loader)
            val_loss = self.validate()
            elapsed = time.time() - start_time
            self.writer.add_scalars('epoch/loss', {'train': train_loss, 'val': val_loss}, epoch + 1)
            self.writer.add_scalar('epoch/lr', self.optimizer.param_groups[0]['lr'], epoch + 1)
            self.writer.add_scalar('epoch/time', elapsed, epoch + 1)
            if self.scheduler:
                self.scheduler.step(val_loss)
            self.save_checkpoint(epoch, train_loss, val_loss)
        self.writer.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    trainer = Trainer(config)
    trainer.train()
