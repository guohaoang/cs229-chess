"""Data loading and preprocessing for chess games from PGN files."""

import io
import os
import zstandard as zstd
import chess
import chess.pgn
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torch
from config import BOARD_CHANNELS, DATASET_DIR, GAMES_PER_FILE
from multiprocessing import Pool
import pickle
from functools import lru_cache


def board_to_tensor(board):
    """
    Convert chess board to tensor representation (optimized).
    
    Args:
        board: chess.Board object
        
    Returns:
        Numpy array of shape (BOARD_CHANNELS, 8, 8)
    """
    tensor = np.zeros((BOARD_CHANNELS, 8, 8), dtype=np.uint8)
    
    # Use board.piece_map() for faster iteration
    for square, piece in board.piece_map().items():
        if piece:
            row, col = divmod(square, 8)
            piece_type = piece.piece_type - 1  # 1-6 -> 0-5
            color_offset = 0 if piece.color == chess.WHITE else 6
            tensor[piece_type + color_offset, row, col] = 1
    
    # Channel 12: side to move (1 = white, 0 = black)
    tensor[12, :, :] = 1 if board.turn == chess.WHITE else 0
    
    return tensor.astype(np.float32)


def move_to_index(move, board):
    """
    Convert chess move to action index.
    
    Simple encoding: from_square * 64 + to_square (ignores promotions for now)
    
    Args:
        move: chess.Move object
        board: chess.Board object
        
    Returns:
        Integer index (0-4095 for basic moves)
    """
    return move.from_square * 64 + move.to_square


def load_pgn_games(pgn_path, max_games=None):
    """
    Load games from a zstandard-compressed PGN file (optimized streaming).
    
    Args:
        pgn_path: Path to .pgn.zst file
        max_games: Maximum number of games to load (None = all)
        
    Returns:
        List of (board_state_numpy, move_index) tuples
    """
    games_data = []
    game_count = 0
    
    try:
        print(f"Decompressing {Path(pgn_path).name}...")
        with open(pgn_path, "rb") as f:
            dctx = zstd.ZstdDecompressor()
            reader = dctx.stream_reader(f, closefd=False)
            pgn_text = reader.read().decode("utf-8", errors="ignore")
            print(f"Parsing games from {Path(pgn_path).name}...")
            pgn_io = io.StringIO(pgn_text)
            
            while True:
                game = chess.pgn.read_game(pgn_io)
                if game is None:
                    break
                
                try:
                    board = game.board()
                    pawn_count = 0
                    
                    # Extract moves from game
                    for move in game.mainline_moves():
                        # Count pawns for filtering
                        pawn_count = len(board.pieces(chess.PAWN, chess.WHITE)) + \
                                    len(board.pieces(chess.PAWN, chess.BLACK))
                        
                        if pawn_count > 2:  # At least some pawns remain
                            board_tensor = board_to_tensor(board)
                            move_idx = move_to_index(move, board)
                            games_data.append((board_tensor, move_idx))
                        
                        board.push(move)
                    
                    game_count += 1
                    if max_games and game_count >= max_games:
                        break
                    
                    if game_count % 500 == 0:
                        print(f"Loaded {game_count} games, {len(games_data)} positions from {Path(pgn_path).name}")
                
                except Exception as e:
                    continue
    
    except Exception as e:
        print(f"Error reading file {pgn_path}: {e}")
    
    print(f"Finished {Path(pgn_path).name}: {game_count} games, {len(games_data)} positions")
    return games_data


class ChessDataset(Dataset):
    """PyTorch Dataset for chess positions with lazy loading."""

    def __init__(self, pgn_files, max_games_per_file=None, transform=None, 
                 use_cache=True, cache_dir="./data_cache", num_workers=4):
        """
        Initialize dataset.
        
        Args:
            pgn_files: List of paths to PGN files
            max_games_per_file: Maximum games to load per file
            transform: Optional data augmentation transform
            use_cache: Cache processed data to disk
            cache_dir: Directory for caching
            num_workers: Number of parallel workers for loading
        """
        self.data = []
        self.transform = transform
        self.use_cache = use_cache
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data with optional parallelization
        for pgn_file in pgn_files:
            cache_path = self.cache_dir / f"{Path(pgn_file).stem}.pkl"
            
            # Try loading from cache
            if use_cache and cache_path.exists():
                print(f"Loading cache from {cache_path.name}...")
                try:
                    with open(cache_path, 'rb') as f:
                        games_data = pickle.load(f)
                    print(f"Loaded {len(games_data)} positions from cache")
                    self.data.extend(games_data)
                    continue
                except Exception as e:
                    print(f"Cache load failed, reprocessing: {e}")
            
            # Load from PGN
            print(f"Loading {pgn_file}...")
            games_data = load_pgn_games(pgn_file, max_games=max_games_per_file)
            
            # Save to cache
            if use_cache:
                try:
                    with open(cache_path, 'wb') as f:
                        pickle.dump(games_data, f)
                    print(f"Cached {len(games_data)} positions to {cache_path.name}")
                except Exception as e:
                    print(f"Cache save failed: {e}")
            
            self.data.extend(games_data)
            print(f"Total positions: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        board_tensor, move_idx = self.data[idx]
        
        # Convert numpy to torch tensor
        if isinstance(board_tensor, np.ndarray):
            board_tensor = torch.from_numpy(board_tensor)
        
        if self.transform:
            board_tensor = self.transform(board_tensor)
        
        return board_tensor, move_idx


def get_dataloaders(pgn_dir, batch_size, num_workers=0, train_split=0.8, 
                   max_games_per_file=None, use_cache=True):
    """
    Create train/val/test dataloaders.
    
    Args:
        pgn_dir: Directory containing PGN files
        batch_size: Batch size for loading
        num_workers: Number of worker processes for DataLoader
        train_split: Fraction of data for training
        max_games_per_file: Max games to load per file
        use_cache: Use cached preprocessed data
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Find all PGN files
    pgn_files = sorted(Path(pgn_dir).glob("*.pgn.zst"))
    
    if not pgn_files:
        raise FileNotFoundError(f"No PGN files found in {pgn_dir}")
    
    print(f"Found {len(pgn_files)} PGN files")
    
    # Load dataset with caching
    dataset = ChessDataset(pgn_files[:5], max_games_per_file=max_games_per_file,
                          use_cache=use_cache)
    
    # Split dataset
    train_size = int(len(dataset) * train_split)
    val_size = int(len(dataset) * (1 - train_split) / 2)
    test_size = len(dataset) - train_size - val_size
    
    train_set, val_set, test_set = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, 
                             num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, 
                           num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers)
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test data loading
    pgn_dir = DATASET_DIR
    train_loader, val_loader, test_loader = get_dataloaders(
        pgn_dir, 
        batch_size=32, 
        max_games_per_file=1000
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Sample batch
    batch = next(iter(train_loader))
    print(f"Batch shape: {batch[0].shape}, {batch[1].shape}")
