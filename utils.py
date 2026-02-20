"""Utility functions for training and evaluation."""

import torch
import torch.nn as nn
from pathlib import Path
import json
from datetime import datetime


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self, name, fmt=':.4f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter:
    """Displays progress of training."""

    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir, name=None):
    """Save model checkpoint."""
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    if name is None:
        name = f"checkpoint_epoch_{epoch}.pt"
    
    checkpoint_path = Path(checkpoint_dir) / name
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)
    
    print(f"Checkpoint saved: {checkpoint_path}")


def load_checkpoint(model, optimizer, checkpoint_path):
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    print(f"Checkpoint loaded from {checkpoint_path}")
    return epoch, loss


def accuracy_topk(output, target, topk=(1, 5)):
    """Compute top-k accuracy."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        
        return res


def log_metrics(epoch, train_loss, val_loss, train_acc, val_acc, log_dir="./logs"):
    """Log training metrics."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    metrics = {
        'epoch': epoch,
        'timestamp': datetime.now().isoformat(),
        'train_loss': float(train_loss),
        'val_loss': float(val_loss),
        'train_acc': float(train_acc),
        'val_acc': float(val_acc),
    }
    
    log_file = Path(log_dir) / "metrics.jsonl"
    with open(log_file, 'a') as f:
        f.write(json.dumps(metrics) + '\n')
    
    return metrics
