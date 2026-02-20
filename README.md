# Chess ML: Predicting Human Move Behavior from Lichess

A machine learning project to train CNN models that learn to predict human move behavior from Lichess standard chess database.

## Project Structure

```
.
├── config.py          # Configuration settings
├── model.py           # CNN model architectures
├── dataset.py         # Data loading and preprocessing
├── utils.py           # Utility functions
├── train.py           # Main training script
├── requirements.txt   # Python dependencies
└── datasets/          # Chess game databases (PGN.zst format)
```

## Setup

1. **Create virtual environment**
```bash
conda create -n chess-ml python=3.10 -y
conda activate chess-ml
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Prepare datasets**
Ensure PGN files are in `datasets/` or `datasets_small/` directories.

## Usage

### Training

Start training with default configuration:
```bash
python train.py
```

Options:
```bash
python train.py --model v2 --batch-size 256 --epochs 20 --lr 0.001
python train.py --model v1 --dataset-dir ./datasets_small --device cuda
```

### Monitoring

Training metrics are logged to `logs/metrics.jsonl`. Checkpoints are saved to `checkpoints/`.

## Model Architectures

### ChessCNN (v1)
- Stacked convolutional layers (64, 128, 256 filters)
- Followed by fully connected layers (512, 256)
- ~2.8M parameters

### ChessCNNv2 (v2)
- Initial conv block + 3 residual-like blocks
- More efficient feature extraction
- ~1.5M parameters

## Data Representation

- **Board state**: 13 channels
  - 12 channels: 6 piece types × 2 colors (white/black)
  - 1 channel: Side to move
- **Move encoding**: From-square × 64 + to-square (simplified; ignores promotions)

## References

- Lichess Database: https://database.lichess.org/
- PyTorch: https://pytorch.org/
- python-chess: https://python-chess.readthedocs.io/
