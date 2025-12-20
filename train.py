"""TrackNet Training Script

Usage:
    python train.py --config config.yaml
"""

import os
import signal
import subprocess
import sys
import time
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model.loss import WeightedBinaryCrossEntropy
from model.tracknet_v2 import TrackNet
from preprocessing.tracknet_dataset import FrameHeatmapDataset


def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)['train']


def setup_ddp():
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        dist.init_process_group('nccl')
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size
    return 0, 0, 1


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


class Trainer:
    def __init__(self, cfg, rank=0, local_rank=0, world_size=1):
        self.cfg = cfg
        self.rank = rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.is_main = rank == 0
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
        if self.world_size > 1:
            return torch.device(f'cuda:{self.local_rank}')
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
            if self.is_main:
                self.writer = SummaryWriter(log_dir=str(self.save_dir / "tensorboard"))
                print(f"Resuming from epoch {self.start_epoch}, step {self.step}")
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.save_dir = Path(self.cfg['out']) / f"{self.cfg['name']}_{timestamp}"
            if self.is_main:
                self.save_dir.mkdir(parents=True, exist_ok=True)
                (self.save_dir / "checkpoints").mkdir(exist_ok=True)
                self.writer = SummaryWriter(log_dir=str(self.save_dir / "tensorboard"))
        if self.world_size > 1:
            dist.barrier()

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

        if self.world_size > 1:
            self.train_sampler = DistributedSampler(train_ds, shuffle=True)
            self.val_sampler = DistributedSampler(val_ds, shuffle=False)
            self.train_loader = DataLoader(
                train_ds, batch_size=self.cfg['batch'], sampler=self.train_sampler,
                num_workers=self.cfg['workers'], pin_memory=True
            )
            self.val_loader = DataLoader(
                val_ds, batch_size=self.cfg['batch'], sampler=self.val_sampler,
                num_workers=self.cfg['workers'], pin_memory=True
            )
        else:
            self.train_sampler = None
            self.val_sampler = None
            self.train_loader = DataLoader(
                train_ds, batch_size=self.cfg['batch'], shuffle=True,
                num_workers=self.cfg['workers'], pin_memory=self.device.type == 'cuda'
            )
            self.val_loader = DataLoader(
                val_ds, batch_size=self.cfg['batch'], shuffle=False,
                num_workers=self.cfg['workers'], pin_memory=self.device.type == 'cuda'
            )

        if self.is_main and not self.checkpoint:
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
        dropout = self.cfg.get('dropout', 0.0)
        self.model = TrackNet(dropout=dropout).to(self.device)
        if self.world_size > 1:
            self.model = DDP(self.model, device_ids=[self.local_rank])
        self.criterion = WeightedBinaryCrossEntropy()
        self.optimizer = self._create_optimizer()
        if self.cfg['scheduler'] == "ReduceLROnPlateau":
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, mode='min', factor=self.cfg['factor'],
                patience=self.cfg['patience'], min_lr=self.cfg['min_lr']
            )
        else:
            self.scheduler = None
        if self.checkpoint:
            state_dict = self.checkpoint['model_state_dict']
            if self.world_size > 1 and not list(state_dict.keys())[0].startswith('module.'):
                state_dict = {f'module.{k}': v for k, v in state_dict.items()}
            elif self.world_size == 1 and list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            self.model.load_state_dict(state_dict)
            if 'optimizer_state_dict' in self.checkpoint:
                self.optimizer.load_state_dict(self.checkpoint['optimizer_state_dict'])
            if self.scheduler and 'scheduler_state_dict' in self.checkpoint:
                self.scheduler.load_state_dict(self.checkpoint['scheduler_state_dict'])

    def save_checkpoint(self, epoch, train_loss, val_loss, is_emergency=False):
        if not self.is_main:
            return None, False
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_state = self.model.module.state_dict() if self.world_size > 1 else self.model.state_dict()
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state,
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
            for inputs, targets in tqdm(self.val_loader, desc="Validating", leave=False, disable=not self.is_main):
                if self.interrupted:
                    break
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
        avg_loss = total_loss / len(self.val_loader)
        if self.world_size > 1:
            loss_tensor = torch.tensor(avg_loss, device=self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            avg_loss = loss_tensor.item()
        return avg_loss

    def train(self):
        self.setup_data()
        self.setup_model()

        if self.is_main:
            print(f"TensorBoard: tensorboard --logdir {self.save_dir / 'tensorboard'}")
            if self.world_size > 1:
                print(f"Training with {self.world_size} GPUs")

        for epoch in range(self.start_epoch, self.cfg['epochs']):
            if self.interrupted:
                break
            if self.train_sampler:
                self.train_sampler.set_epoch(epoch)
            start_time = time.time()
            self.model.train()
            total_loss = 0.0
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.cfg['epochs']}",
                        leave=False, disable=not self.is_main)
            for batch_idx, (inputs, targets) in enumerate(pbar):
                if self.interrupted:
                    pbar.close()
                    val_loss = self.validate()
                    self.save_checkpoint(epoch, total_loss / (batch_idx + 1), val_loss, True)
                    if self.is_main:
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
                if self.is_main:
                    current_lr = self._calculate_effective_lr()
                    self.writer.add_scalar('batch/loss', batch_loss, self.step)
                    self.writer.add_scalar('batch/lr', current_lr, self.step)
                pbar.set_postfix({'loss': f'{batch_loss:.6f}'})
            pbar.close()
            train_loss = total_loss / len(self.train_loader)
            val_loss = self.validate()
            elapsed = time.time() - start_time
            if self.is_main:
                self.writer.add_scalars('epoch/loss', {'train': train_loss, 'val': val_loss}, epoch + 1)
                self.writer.add_scalar('epoch/lr', self.optimizer.param_groups[0]['lr'], epoch + 1)
                self.writer.add_scalar('epoch/time', elapsed, epoch + 1)
            if self.scheduler:
                self.scheduler.step(val_loss)
            self.save_checkpoint(epoch, train_loss, val_loss)
            if self.world_size > 1:
                dist.barrier()
        if self.is_main:
            self.writer.close()


def main():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        full_cfg = yaml.safe_load(f)

    train_cfg = full_cfg['train']
    gpus = train_cfg.get('gpus', 1)

    if gpus > 1 and 'RANK' not in os.environ:
        cmd = [
            'torchrun',
            f'--nproc_per_node={gpus}',
            sys.argv[0],
            '--config', args.config
        ]
        subprocess.run(cmd)
    else:
        rank, local_rank, world_size = setup_ddp()
        try:
            trainer = Trainer(train_cfg, rank, local_rank, world_size)
            trainer.train()
        finally:
            cleanup_ddp()


if __name__ == "__main__":
    main()
