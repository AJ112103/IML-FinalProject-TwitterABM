#!/usr/bin/env python3
"""
Preprocess the Twitter retweet analysis data:
- Clean the CSV data
- Group retweets to form cascades
- Generate time series of retweet counts
- Create synthetic network data for ABM
- Classify tweets as viral or non-viral
"""

import os
import sys
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import yaml
from datetime import datetime
import random

# Setup logging
os.makedirs(os.path.join(os.path.dirname(__file__), '..', '..', 'logs'), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(os.path.dirname(__file__), '..', '..', 'logs', 'preprocess.log'), mode='w')
    ]
)
logger = logging.getLogger('preprocess')

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def preprocess_data(config):
    """Preprocess the raw tweet data"""
    logger.info("Starting data preprocessing")
    
    # Create processed data directory if it doesn't exist
    os.makedirs(config['data']['processed_dir'], exist_ok=True)
    
    # Load the raw data
    raw_data_path = os.path.join(config['data']['raw_dir'], config['data']['filename'])
    logger.info(f"Loading raw data from {raw_data_path}")
    
    try:
        df = pd.read_csv(raw_data_path)
        logger.info(f"Loaded {len(df)} records")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return False
    
    # Basic cleaning
    logger.info("Performing basic data cleaning")
    
    # Handle missing values
    for col in df.columns:
        if df[col].isna().sum() > 0:
            logger.info(f"Filling missing values in {col}")
            if df[col].dtype in [np.int64, np.float64]:
                df[col].fillna(0, inplace=True)
            else:
                df[col].fillna('unknown', inplace=True)
    
    # Feature engineering
    logger.info("Creating additional features")
    
    # Normalize numerical columns
    numerical_cols = ['reach', 'retweetcount', 'likes', 'klout']
    for col in numerical_cols:
        df[f'{col}_norm'] = df[col] / df[col].max()
    
    # Create virality classifications using different metrics
    # 1. Based on retweet count threshold (90th percentile)
    rt_threshold = df.retweetcount.quantile(0.9)  # 90th percentile (~16)
    df['is_viral_by_rt'] = (df.retweetcount > rt_threshold).astype(int)
    
    # 2. Multi-class virality classification
    def classify_virality(rt_count):
        if rt_count <= 5:
            return 0  # Not viral
        elif rt_count <= 20:
            return 1  # Moderately viral
        else:
            return 2  # Highly viral
    
    df['virality_class'] = df.retweetcount.apply(classify_virality)
    
    # 3. Composite virality score (weighted RT, reach, likes)
    df['virality_score'] = (
        0.5 * df.retweetcount_norm + 
        0.3 * df.reach_norm + 
        0.2 * df.likes_norm
    )
    score_threshold = df.virality_score.quantile(0.9)
    df['is_viral_by_score'] = (df.virality_score > score_threshold).astype(int)
    
    # Apply chosen classification method based on config
    virality_method = config.get('virality_method', 'retweet_threshold')
    if virality_method == 'retweet_threshold':
        logger.info(f"Using retweet threshold ({rt_threshold}) for virality")
        df['is_viral'] = df['is_viral_by_rt']
    elif virality_method == 'composite_score':
        logger.info(f"Using composite score ({score_threshold:.4f}) for virality")
        df['is_viral'] = df['is_viral_by_score']
    elif virality_method == 'multi_class':
        logger.info("Using multi-class approach for virality")
        df['is_viral'] = (df['virality_class'] > 0).astype(int)
    else:
        logger.info(f"Unknown virality method {virality_method}, defaulting to retweet threshold")
        df['is_viral'] = df['is_viral_by_rt']
    
    # Log virality statistics
    viral_count = df['is_viral'].sum()
    logger.info(f"Classified {viral_count} tweets as viral ({viral_count/len(df)*100:.2f}%)")
    
    # Generate time-series features
    logger.info("Generating time-series features")
    
    # Group by day and hour (without trying to create an actual datetime)
    time_series = df.groupby(['weekday', 'hour']).agg({
        'retweetcount': 'sum',
        'is_viral': 'sum',
        'reach': 'mean',
        'likes': 'sum'
    }).reset_index()
    
    # Add time_step field for easier analysis
    time_series['time_step'] = range(len(time_series))
    
    # Create synthetic network data for ABM
    logger.info("Creating synthetic network data for ABM")
    
    # Generate user networks based on klout scores and followers (estimated by reach)
    unique_users = len(df['locationid'].unique())
    logger.info(f"Found {unique_users} unique user locations")
    
    # Create a user network dataset with influence metrics
    user_data = df.groupby('locationid').agg({
        'klout': 'mean',
        'reach': 'mean',
        'retweetcount': 'mean',
        'likes': 'mean',
        'sentiment': 'mean',
        'isreshare': 'mean'  # Proportion of reshares
    }).reset_index()
    
    # Scale user influence metrics
    user_data['influence'] = user_data['klout'] / 100.0  # Scale to 0-1
    user_data['susceptibility'] = 0.3 + 0.4 * user_data['isreshare']  # Base susceptibility plus reshare tendency
    
    # Create follower network (simplified)
    # Estimate followers from reach (simple heuristic)
    user_data['followers'] = np.round(user_data['reach'] * 0.1).astype(int)  # Rough estimate
    
    # Save processed data
    logger.info("Saving processed data")
    
    # Main dataset with virality labels
    processed_path = os.path.join(config['data']['processed_dir'], 'tweets_processed.csv')
    df.to_csv(processed_path, index=False)
    logger.info(f"Saved processed tweet data to {processed_path}")
    
    # Time series dataset
    timeseries_path = os.path.join(config['data']['processed_dir'], 'tweet_timeseries.csv')
    time_series.to_csv(timeseries_path, index=False)
    logger.info(f"Saved time series data to {timeseries_path}")
    
    # User network dataset
    user_path = os.path.join(config['data']['processed_dir'], 'user_network.csv')
    user_data.to_csv(user_path, index=False)
    logger.info(f"Saved user network data to {user_path}")
    
    # Create train/val/test splits
    logger.info("Creating dataset splits")
    
    # Shuffle data
    df_shuffled = df.sample(frac=1, random_state=config['random_seed'])
    
    # Split based on config ratios
    train_ratio = config['data']['train_ratio']
    val_ratio = config['data']['val_ratio']
    
    train_size = int(len(df_shuffled) * train_ratio)
    val_size = int(len(df_shuffled) * val_ratio)
    
    train_data = df_shuffled[:train_size]
    val_data = df_shuffled[train_size:train_size+val_size]
    test_data = df_shuffled[train_size+val_size:]
    
    # Save splits
    train_path = os.path.join(config['data']['processed_dir'], 'train.csv')
    val_path = os.path.join(config['data']['processed_dir'], 'val.csv')
    test_path = os.path.join(config['data']['processed_dir'], 'test.csv')
    
    train_data.to_csv(train_path, index=False)
    val_data.to_csv(val_path, index=False)
    test_data.to_csv(test_path, index=False)
    
    logger.info(f"Saved {len(train_data)} training samples, {len(val_data)} validation samples, {len(test_data)} test samples")
    
    logger.info("Preprocessing complete")
    return True

def main():
    parser = argparse.ArgumentParser(description='Preprocess Twitter retweet data')
    parser.add_argument('--config', type=str, default=os.path.join(os.path.dirname(__file__), '..', '..', 'configs', 'default.yaml'), help='Path to the config file')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Process data
    success = preprocess_data(config)
    
    if success:
        logger.info("Preprocessing completed successfully")
    else:
        logger.error("Preprocessing failed")
        sys.exit(1)

if __name__ == "__main__":
    main() 