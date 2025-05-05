#!/usr/bin/env python3
"""
Main script to run the entire pipeline for the Twitter virality prediction project.
"""

import os
import sys
import argparse
import logging
import subprocess
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/run.log', mode='w')
    ]
)
logger = logging.getLogger('run')

# Create required directories
os.makedirs('logs', exist_ok=True)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run the entire pipeline')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to configuration file')
    parser.add_argument('--skip_download', action='store_true',
                        help='Skip data download step')
    parser.add_argument('--skip_preprocess', action='store_true',
                        help='Skip data preprocessing step')
    parser.add_argument('--models', type=str, default='cnn,transformer,mlp,svm',
                        help='Comma-separated list of models to train and evaluate')
    parser.add_argument('--epochs', type=int, default=25,
                        help='Number of epochs for training')
    return parser.parse_args()

def run_command(command, description):
    """Run a shell command and log output."""
    logger.info(f"Running: {description}")
    logger.info(f"Command: {command}")
    
    start_time = time.time()
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    elapsed_time = time.time() - start_time
    
    logger.info(f"Exit code: {result.returncode}")
    logger.info(f"Completed in {elapsed_time:.2f} seconds")
    
    if result.returncode != 0:
        logger.error(f"Error output: {result.stderr}")
        logger.warning(f"Command failed: {description}")
    else:
        logger.info(f"Output: {result.stdout[:500]}...")  # Show truncated output
    
    return result.returncode == 0

def download_data(config_path):
    """Download the data."""
    return run_command(
        f"python src/data/download.py --config {config_path}",
        "Data download"
    )

def preprocess_data(config_path):
    """Preprocess the data."""
    return run_command(
        f"python src/data/preprocess.py --config {config_path}",
        "Data preprocessing"
    )

def train_model(model, config_path, epochs):
    """Train a model."""
    return run_command(
        f"python src/train.py --model {model} --config {config_path} --epochs {epochs}",
        f"Training {model} model"
    )

def evaluate_model(model, config_path):
    """Evaluate a model."""
    return run_command(
        f"python src/evaluate.py --model {model} --config {config_path} --ckpt results/{model}_best.npz",
        f"Evaluating {model} model"
    )

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    logger.info("Starting pipeline")
    
    # Download data
    if not args.skip_download:
        success = download_data(args.config)
        if not success:
            logger.error("Data download failed. Exiting.")
            sys.exit(1)
    else:
        logger.info("Skipping data download")
    
    # Preprocess data
    if not args.skip_preprocess:
        success = preprocess_data(args.config)
        if not success:
            logger.error("Data preprocessing failed. Exiting.")
            sys.exit(1)
    else:
        logger.info("Skipping data preprocessing")
    
    # Parse models
    models = args.models.split(',')
    logger.info(f"Models to train and evaluate: {models}")
    
    # Train and evaluate each model
    for model in models:
        # Train model
        success = train_model(model, args.config, args.epochs)
        if not success:
            logger.error(f"Training {model} failed. Skipping evaluation.")
            continue
        
        # Evaluate model
        success = evaluate_model(model, args.config)
        if not success:
            logger.error(f"Evaluating {model} failed.")
    
    logger.info("Pipeline completed")

if __name__ == "__main__":
    main() 