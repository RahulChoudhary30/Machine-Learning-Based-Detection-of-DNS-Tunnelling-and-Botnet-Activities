import os
import glob
import pandas as pd
import numpy as np
import random
from src.utils import setup_logger

logger = setup_logger()

def load_and_split_data(config):
    """
    Loads parquet files from the data directory and splits them into train and test sets.
    Supports 'file' or 'row' based splitting configured via split_strategy.
    Returns:
        train_df, test_df
    """
    data_dir = config["paths"]["data_dir"]
    max_files = config["data"].get("max_files", None)
    train_ratio = config["data"].get("train_split", 0.8)
    split_strategy = config["data"].get("split_strategy", "file")
    
    # Get all parquet files
    all_files = glob.glob(os.path.join(data_dir, "*.parquet"))
    if not all_files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")
        
    logger.info(f"Found {len(all_files)} parquet files.")
    
    if max_files and max_files < len(all_files):
        logger.info(f"Using only {max_files} files as configured.")
        all_files = all_files[:max_files]
        
    random.seed(config["training"].get("random_state", 42))
    random.shuffle(all_files)
    
    if split_strategy == "row" or len(all_files) == 1:
        if len(all_files) == 1 and split_strategy != "row":
            logger.warning("Only 1 file provided. Breaking file-based split rule and splitting rows.")
        logger.info(f"Loading {len(all_files)} files for row-based split...")
        merged_df = _read_files(all_files)
        
        logger.info("Performing row-based train-test split...")
        from sklearn.model_selection import train_test_split
        # Stratify by temporary binary label to ensure balanced classes in train/test and avoid single-member class errors
        target_col = config["data"].get("target_col", "label")
        if target_col in merged_df.columns:
            stratify_col = merged_df[target_col].astype(str).str.contains('Botnet', case=False, na=False).astype(int)
        else:
            stratify_col = None
        
        train_df, test_df = train_test_split(
            merged_df, 
            train_size=train_ratio, 
            random_state=config["training"].get("random_state", 42),
            stratify=stratify_col
        )
        return train_df, test_df

    else:
        # File-based split
        split_idx = int(len(all_files) * train_ratio)
        # Ensure at least 1 file in test if possible
        if split_idx == len(all_files) and len(all_files) > 1:
            split_idx -= 1
        elif split_idx == 0 and len(all_files) > 1:
            split_idx = 1
            
        train_files = all_files[:split_idx]
        test_files = all_files[split_idx:]
        
        logger.info(f"Training on {len(train_files)} files: {[os.path.basename(f) for f in train_files]}")
        logger.info(f"Testing on {len(test_files)} files: {[os.path.basename(f) for f in test_files]}")
        
        train_df = _read_files(train_files)
        test_df = _read_files(test_files) if test_files else pd.DataFrame()
            
        return train_df, test_df

def _read_files(file_paths):
    """Memory-optimized reading of parquet files."""
    dfs = []
    columns_to_read = None # Read all columns
    
    for f in file_paths:
        logger.info(f"Loading {os.path.basename(f)}...")
        try:
            # Using fastparquet engine or pyarrow. Pandas handles memory efficiently with parquet
            df = pd.read_parquet(f, engine="pyarrow")
            dfs.append(df)
        except Exception as e:
            logger.error(f"Failed to read {f}: {str(e)}")
            
    if not dfs:
        raise ValueError("No data loaded successfully.")
        
    merged_df = pd.concat(dfs, ignore_index=True)
    
    # Memory optimization: Downcast numeric columns
    for col in merged_df.select_dtypes(include=['float64']).columns:
        merged_df[col] = pd.to_numeric(merged_df[col], downcast='float')
    for col in merged_df.select_dtypes(include=['int64']).columns:
        merged_df[col] = pd.to_numeric(merged_df[col], downcast='integer')
        
    logger.info(f"Loaded {len(merged_df)} rows. Memory usage: is {merged_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    return merged_df
