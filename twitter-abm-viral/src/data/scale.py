#!/usr/bin/env python3
"""
Feature scaling utility for SVM model training.
For troubleshooting SVM convergence issues.
"""

import os
import sys
import argparse
import numpy as np
import logging
from pathlib import Path
import yaml

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/scale.log', mode='w')
    ]
)
logger = logging.getLogger('scale')

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Scale features for SVM training')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to configuration file')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save scaled features (defaults to original file with _scaled suffix)')
    return parser.parse_args()

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def scale_features(data_path, output_path=None):
    """
    Scale features to [-1, 1] range for SVM.
    
    Args:
        data_path (str): Path to data file (.npy)
        output_path (str, optional): Path to save scaled features
    
    Returns:
        np.ndarray: Scaled features
    """
    # Load data
    X = np.load(data_path)
    logger.info(f"Loaded data with shape {X.shape} from {data_path}")
    
    # Scale features to [-1, 1]
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X_range = X_max - X_min
    
    # Handle zero range (constant features)
    X_range[X_range == 0] = 1.0
    
    # Scale to [-1, 1]
    X_scaled = 2 * (X - X_min) / X_range - 1
    
    # Save scaling parameters
    scaling_params = {
        'min': X_min,
        'max': X_max,
        'range': X_range
    }
    
    # Save scaled features
    if output_path is None:
        # Create default output path
        path_obj = Path(data_path)
        output_path = str(path_obj.parent / f"{path_obj.stem}_scaled{path_obj.suffix}")
    
    np.save(output_path, X_scaled)
    logger.info(f"Saved scaled features to {output_path}")
    
    # Save scaling parameters
    params_path = str(Path(output_path).parent / f"{Path(output_path).stem}_params.npz")
    np.savez(params_path, **scaling_params)
    logger.info(f"Saved scaling parameters to {params_path}")
    
    return X_scaled, scaling_params

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Get data path
    processed_path = config['data']['processed_path']
    data_path = os.path.join(processed_path, 'hourly_counts.npy')
    
    # Scale features
    scale_features(data_path, args.output)
    
    logger.info("Feature scaling completed")

if __name__ == "__main__":
    main() 