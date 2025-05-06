#!/usr/bin/env python3

import os
import sys
import logging
import argparse
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
import time
import csv

from src.models.cnn import CNN
from src.models.pca_ica import DimensionReducer
from src.models.transformer import TransformerModel
from src.models.classical import MLP, SVM

os.makedirs(os.path.join(os.path.dirname(__file__), '..', 'logs'), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(os.path.dirname(__file__), '..', 'logs', 'train.log'), mode='a')
    ]
)
logger = logging.getLogger('train')

os.makedirs(os.path.join(os.path.dirname(__file__), '..', 'logs'), exist_ok=True)
os.makedirs(os.path.join(os.path.dirname(__file__), '..', 'results'), exist_ok=True)
os.makedirs(os.path.join(os.path.dirname(__file__), '..', 'models_saved'), exist_ok=True)

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_data(config, dataset_type='processed'):
    logger.info(f"Loading {dataset_type} dataset")
    
    processed_dir = config['data']['processed_dir']
    
    features = ['reach', 'retweetcount', 'likes', 'klout', 'sentiment', 
                'isreshare', 'klout_norm', 'reach_norm', 'retweetcount_norm',
                'likes_norm', 'virality_score']
    
    if dataset_type == 'processed':
        train_path = os.path.join(processed_dir, 'train.csv')
        val_path = os.path.join(processed_dir, 'val.csv')
        
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        
        target = 'is_viral'
        
        available_features = [f for f in features if f in train_df.columns]
        
        logger.info(f"Using features: {available_features}")
        logger.info(f"Using target: {target}")
        
        X_train = train_df[available_features].values
        y_train = train_df[target].values
        
        X_val = val_df[available_features].values
        y_val = val_df[target].values
    
    elif dataset_type == 'simulated':
        sim_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'training_data.csv')
        
        if not os.path.exists(sim_path):
            logger.error(f"Simulated data not found at {sim_path}")
            return None
        
        sim_df = pd.read_csv(sim_path)
        
        sim_df = sim_df.sample(frac=1, random_state=config['random_seed'])
        split_idx = int(len(sim_df) * 0.8)
        
        train_df = sim_df[:split_idx]
        val_df = sim_df[split_idx:]
        
        sim_features = ['total_retweets', 'peak_volume', 'peak_time', 'early_retweets',
                        'tweet_quality', 'sentiment', 'seed_fraction']
        
        available_features = [f for f in sim_features if f in train_df.columns]
        
        logger.info(f"Using features: {available_features}")
        logger.info(f"Using target: is_viral")
        
        X_train = train_df[available_features].values
        y_train = train_df['is_viral'].values
        
        X_val = val_df[available_features].values
        y_val = val_df['is_viral'].values
    
    elif dataset_type == 'both':
        real_train_path = os.path.join(processed_dir, 'train.csv')
        real_val_path = os.path.join(processed_dir, 'val.csv')
        sim_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'training_data.csv')
        
        if not os.path.exists(real_train_path) or not os.path.exists(sim_path):
            logger.error("Both real and simulated datasets must exist")
            return None
        
        real_train_df = pd.read_csv(real_train_path)
        real_val_df = pd.read_csv(real_val_path)
        sim_df = pd.read_csv(sim_path)
        
        sim_df = sim_df.sample(frac=1, random_state=config['random_seed'])
        split_idx = int(len(sim_df) * 0.8)
        sim_train_df = sim_df[:split_idx]
        sim_val_df = sim_df[split_idx:]
        
        real_features = [f for f in features if f in real_train_df.columns]
        
        real_train_df['peak_volume'] = real_train_df['retweetcount']
        real_train_df['peak_time'] = np.random.randint(0, 12, size=len(real_train_df))
        real_train_df['early_retweets'] = real_train_df['retweetcount'] * 0.7
        real_train_df['tweet_quality'] = real_train_df['klout_norm']
        real_train_df['seed_fraction'] = 0.01
        
        real_val_df['peak_volume'] = real_val_df['retweetcount']
        real_val_df['peak_time'] = np.random.randint(0, 12, size=len(real_val_df))
        real_val_df['early_retweets'] = real_val_df['retweetcount'] * 0.7
        real_val_df['tweet_quality'] = real_val_df['klout_norm']
        real_val_df['seed_fraction'] = 0.01
        
        combined_features = ['retweetcount', 'peak_volume', 'peak_time', 'early_retweets',
                            'sentiment', 'tweet_quality', 'seed_fraction']
        
        available_features = [f for f in combined_features if f in real_train_df.columns and f in sim_train_df.columns]
        
        logger.info(f"Using combined features: {available_features}")
        
        X_train_real = real_train_df[available_features].values
        y_train_real = real_train_df['is_viral'].values
        
        X_train_sim = sim_train_df[available_features].values
        y_train_sim = sim_train_df['is_viral'].values
        
        X_val_real = real_val_df[available_features].values
        y_val_real = real_val_df['is_viral'].values
        
        X_val_sim = sim_val_df[available_features].values
        y_val_sim = sim_val_df['is_viral'].values
        
        X_train = np.vstack([X_train_real, X_train_sim])
        y_train = np.concatenate([y_train_real, y_train_sim])
        
        X_val = np.vstack([X_val_real, X_val_sim])
        y_val = np.concatenate([y_val_real, y_val_sim])
    
    else:
        logger.error(f"Unknown dataset type: {dataset_type}")
        return None
    
    logger.info(f"Data loaded: X_train: {X_train.shape}, y_train: {y_train.shape}, " +
                f"X_val: {X_val.shape}, y_val: {y_val.shape}")
    
    train_pos = np.mean(y_train) * 100
    val_pos = np.mean(y_val) * 100
    logger.info(f"Class distribution - Train: {train_pos:.2f}% viral, Val: {val_pos:.2f}% viral")
    
    return X_train, y_train, X_val, y_val

def train_model(model_name, X_train, y_train, X_val, y_val, config, epochs=None):
    logger.info(f"Training {model_name} model")
    
    if epochs is not None:
        training_epochs = epochs
    else:
        training_epochs = config['training']['epochs']
    
    start_time = time.time()
    model = None
    
    if model_name.lower() == 'cnn':
        input_shape = X_train.shape[1]
        
        X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], 1, 1)
        X_val_reshaped = X_val.reshape(X_val.shape[0], X_val.shape[1], 1, 1)
        
        cnn_config = config['models']['cnn']
        model = CNN(
            config=config
        )
        
        metrics = model.train(
            X_train_reshaped, y_train, 
            X_val=X_val_reshaped, y_val=y_val,
            epochs=training_epochs,
            batch_size=config['training']['batch_size']
        )
    
    elif model_name.lower() == 'transformer':
        X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_val_reshaped = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
        
        model = TransformerModel(
            config=config
        )
        
        metrics = model.train(
            X_train_reshaped, y_train, 
            X_val=X_val_reshaped, y_val=y_val,
            epochs=training_epochs,
            batch_size=config['training']['batch_size']
        )
    
    elif model_name.lower() == 'mlp':
        model = MLP(
            input_shape=X_train.shape[1],
            hidden_layers=config['models']['mlp']['hidden_layers'],
            dropout_rate=config['models']['mlp']['dropout_rate'],
            activation=config['models']['mlp']['activation']
        )
        
        metrics = model.fit(
            X_train, y_train, 
            validation_data=(X_val, y_val),
            batch_size=config['training']['batch_size'],
            learning_rate=config['training']['learning_rate'],
            epochs=training_epochs,
            early_stopping_patience=config['training']['early_stopping_patience']
        )
    
    elif model_name.lower() == 'svm':
        model = SVM(
            config=config
        )
        
        metrics = model.fit(X_train, y_train, X_val, y_val)
    
    elif model_name.lower() in ['pca', 'ica']:
        model = DimensionReducer(
            n_components=config['models']['dimension_reduction']['n_components'],
            algorithm=model_name.lower(),
            whiten=config['models']['dimension_reduction']['whiten']
        )
        
        X_train_reduced = model.fit_transform(X_train)
        X_val_reduced = model.transform(X_val)
        
        mlp_model = MLP(
            input_shape=X_train_reduced.shape[1],
            hidden_layers=[64, 32],
            dropout_rate=0.3
        )
        
        metrics = mlp_model.fit(
            X_train_reduced, y_train, 
            validation_data=(X_val_reduced, y_val),
            batch_size=config['training']['batch_size'],
            learning_rate=config['training']['learning_rate'],
            epochs=training_epochs,
            early_stopping_patience=config['training']['early_stopping_patience']
        )
    
    else:
        logger.error(f"Unknown model: {model_name}")
        return None, None
    
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds")
    
    return model, metrics

def save_results(model_name, metrics, config, dataset_type):
    os.makedirs(os.path.join(os.path.dirname(__file__), '..', 'results'), exist_ok=True)
    
    results_path = os.path.join(os.path.join(os.path.dirname(__file__), '..', 'results'), f'{model_name}_{dataset_type}_results.csv')
    
    with open(results_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        writer.writerow(['epoch', 'train_loss', 'train_accuracy', 'val_loss', 'val_accuracy'])
        
        for i, metric in enumerate(metrics):
            writer.writerow([i+1, metric['train_loss'], metric['train_accuracy'], 
                             metric['val_loss'], metric['val_accuracy']])
    
    logger.info(f"Results saved to {results_path}")
    
    config_path = os.path.join(os.path.join(os.path.dirname(__file__), '..', 'results'), f'{model_name}_{dataset_type}_config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    logger.info(f"Configuration saved to {config_path}")

def save_model(model, model_name, dataset_type):
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models_saved')
    os.makedirs(models_dir, exist_ok=True)
    
    model_path = os.path.join(models_dir, f'{model_name}_{dataset_type}_model.npz')
    model.save(model_path)
    
    logger.info(f"Model saved to {model_path}")
    
    best_path = os.path.join(os.path.dirname(__file__), '..', 'results', f'{model_name}_best.npz')
    model.save(best_path)
    
    logger.info(f"Model also saved to {best_path}")

def parse_args():
    parser = argparse.ArgumentParser(description='Train tweet virality prediction models')
    parser.add_argument('--config', type=str, default=os.path.join(os.path.dirname(__file__), '..', 'configs', 'default.yaml'), help='Path to config file')
    parser.add_argument('--model', type=str, required=True, choices=['cnn', 'transformer', 'mlp', 'svm', 'pca', 'ica'], 
                        help='Model to train')
    parser.add_argument('--data', type=str, default='processed', choices=['processed', 'simulated', 'both'],
                        help='Dataset to use for training')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs to train for')
    return parser.parse_args()

def main():
    args = parse_args()
    
    config = load_config(args.config)
    
    np.random.seed(config['random_seed'])
    
    data = load_data(config, dataset_type=args.data)
    if data is None:
        logger.error("Failed to load data")
        sys.exit(1)
    
    X_train, y_train, X_val, y_val = data
    
    model, metrics = train_model(args.model, X_train, y_train, X_val, y_val, config, args.epochs)
    if model is None:
        logger.error("Failed to train model")
        sys.exit(1)
    
    if metrics is not None:
        save_results(args.model, metrics, config, args.data)
    
    save_model(model, args.model, args.data)
    
    logger.info(f"Training of {args.model} model completed successfully")

if __name__ == "__main__":
    main() 