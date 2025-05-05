#!/usr/bin/env python3
"""
Evaluation script for tweet virality prediction models.
Computes metrics and generates figures.
"""

import os
import sys
import logging
import argparse
import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, roc_curve, confusion_matrix,
    average_precision_score
)
from sklearn.model_selection import train_test_split
import time

# Import models
from src.models.cnn import CNN
from src.models.pca_ica import DimensionReducer
from src.models.transformer import TransformerModel
from src.models.classical import MLP, SVM

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/evaluate.log', mode='a')
    ]
)
logger = logging.getLogger('evaluate')

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)
os.makedirs('reports/figures', exist_ok=True)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate tweet virality prediction model')
    parser.add_argument('--model', type=str, required=True,
                        choices=['cnn', 'pca', 'ica', 'transformer', 'mlp', 'svm'],
                        help='Model type to evaluate')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to configuration file')
    parser.add_argument('--ckpt', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--bootstrap', action='store_true',
                        help='Use bootstrap to compute confidence intervals')
    parser.add_argument('--bootstrap_samples', type=int, default=1000,
                        help='Number of bootstrap samples')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility')
    return parser.parse_args()

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_data(config, threshold=None):
    """
    Load training and validation data.
    
    Args:
        config (dict): Configuration dictionary
        threshold (float, optional): Threshold for binary classification
    
    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    processed_path = config['data']['processed_path']
    random_seed = config['data']['random_seed']
    
    # Load data depending on file availability
    try:
        # First, try loading CSV files (tabular data) for SVM and simple models
        try:
            train_df = pd.read_csv(f"{processed_path}/train.csv")
            val_df = pd.read_csv(f"{processed_path}/val.csv")
            test_df = pd.read_csv(f"{processed_path}/test.csv")
            
            # Features and target
            target = 'is_viral'
            features = ['reach', 'retweetcount', 'likes', 'klout', 'sentiment', 'isreshare', 'virality_score']
            available_features = [f for f in features if f in train_df.columns]
            
            # Create feature matrices and target vectors
            X_train = train_df[available_features].values
            y_train = train_df[target].values.reshape(-1, 1)
            
            X_val = val_df[available_features].values
            y_val = val_df[target].values.reshape(-1, 1)
            
            X_test = test_df[available_features].values
            y_test = test_df[target].values.reshape(-1, 1)
            
            logger.info(f"Loaded tabular data from CSV files. Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
            
            return X_train, y_train, X_val, y_val, X_test, y_test
            
        except FileNotFoundError:
            # If CSV files not found, fall back to hourly_counts for time series models
            logger.info("CSV files not found, loading time series data")
            
            # Load hourly counts (time series data)
            hourly_counts = np.load(f"{processed_path}/hourly_counts.npy")
            
            # Load metadata
            metadata_df = pd.read_csv(f"{processed_path}/cascade_metadata.csv")
            
            logger.info(f"Loaded {len(hourly_counts)} cascades from {processed_path}")
            
            # Create labels (binary: viral or not viral)
            if threshold is None:
                # Use median as threshold if not specified
                threshold = np.median(metadata_df['total_retweets'])
            
            y = (metadata_df['total_retweets'] > threshold).astype(int).values.reshape(-1, 1)
            
            # Split data
            train_size = config['data']['train_size']
            val_size = config['data']['val_size']
            test_size = config['data']['test_size']
            
            # Ensure splits sum to 1
            total = train_size + val_size + test_size
            train_size /= total
            val_size /= total
            test_size /= total
            
            # Set random seed for reproducibility
            np.random.seed(random_seed)
            
            # Shuffle indices
            n_samples = len(hourly_counts)
            indices = np.random.permutation(n_samples)
            
            # Calculate split points
            train_end = int(train_size * n_samples)
            val_end = train_end + int(val_size * n_samples)
            
            # Split data
            train_indices = indices[:train_end]
            val_indices = indices[train_end:val_end]
            test_indices = indices[val_end:]
            
            X_train = hourly_counts[train_indices]
            y_train = y[train_indices]
            
            X_val = hourly_counts[val_indices]
            y_val = y[val_indices]
            
            X_test = hourly_counts[test_indices]
            y_test = y[test_indices]
            
            logger.info(f"Data split: {len(X_train)} train, {len(X_val)} validation, {len(X_test)} test")
            
            return X_train, y_train, X_val, y_val, X_test, y_test
    
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        sys.exit(1)

def get_model(model_type, config, checkpoint_path):
    """
    Initialize and load model.
    
    Args:
        model_type (str): Model type
        config (dict): Configuration dictionary
        checkpoint_path (str): Path to model checkpoint
    
    Returns:
        object: Loaded model
    """
    # Initialize model
    if model_type == 'cnn':
        model = CNN(config)
    elif model_type == 'pca':
        model = DimensionReducer(reducer='pca', config=config)
    elif model_type == 'ica':
        model = DimensionReducer(reducer='ica', config=config)
    elif model_type == 'transformer':
        model = TransformerModel(config)
    elif model_type == 'mlp':
        model = MLP(config)
    elif model_type == 'svm':
        model = SVM(config=config)
    else:
        logger.error(f"Unknown model type: {model_type}")
        sys.exit(1)
    
    # Load model weights
    model.load(checkpoint_path)
    logger.info(f"Loaded model from {checkpoint_path}")
    
    return model

def predict(model, model_type, X):
    """
    Generate predictions for a model.
    
    Args:
        model: Model to use
        model_type (str): Model type
        X (np.ndarray): Input data
    
    Returns:
        np.ndarray: Model predictions
    """
    if model_type == 'svm':
        # SVM returns probabilities directly
        y_pred_prob = model.predict(X)
        y_pred_class = (y_pred_prob > 0.5).astype(int)
        
        return y_pred_class, y_pred_prob
    else:
        # Other models return probabilities
        y_pred_prob = model.predict(X)
        y_pred_class = (y_pred_prob > 0.5).astype(int)
        
        return y_pred_class, y_pred_prob

def compute_metrics(y_true, y_pred_class, y_pred_prob):
    """
    Compute evaluation metrics.
    
    Args:
        y_true (np.ndarray): True labels
        y_pred_class (np.ndarray): Predicted classes
        y_pred_prob (np.ndarray): Predicted probabilities
    
    Returns:
        dict: Metrics
    """
    # Ensure labels are flattened
    y_true = y_true.flatten()
    y_pred_class = y_pred_class.flatten()
    y_pred_prob = y_pred_prob.flatten()
    
    # Compute metrics
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred_class),
        'precision': precision_score(y_true, y_pred_class),
        'recall': recall_score(y_true, y_pred_class),
        'f1': f1_score(y_true, y_pred_class),
        'roc_auc': roc_auc_score(y_true, y_pred_prob),
        'pr_auc': average_precision_score(y_true, y_pred_prob)
    }
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred_class)
    metrics['confusion_matrix'] = cm
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    metrics['roc_curve'] = (fpr, tpr)
    
    # PR curve
    precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
    metrics['pr_curve'] = (precision, recall)
    
    return metrics

def bootstrap_metrics(y_true, y_pred_class, y_pred_prob, n_samples=1000, confidence=0.95):
    """
    Compute bootstrap confidence intervals for metrics.
    
    Args:
        y_true (np.ndarray): True labels
        y_pred_class (np.ndarray): Predicted classes
        y_pred_prob (np.ndarray): Predicted probabilities
        n_samples (int): Number of bootstrap samples
        confidence (float): Confidence level
    
    Returns:
        dict: Metrics with confidence intervals
    """
    # Flatten arrays
    y_true = y_true.flatten()
    y_pred_class = y_pred_class.flatten()
    y_pred_prob = y_pred_prob.flatten()
    
    # Initialize arrays for bootstrap samples
    accuracy_samples = np.zeros(n_samples)
    precision_samples = np.zeros(n_samples)
    recall_samples = np.zeros(n_samples)
    f1_samples = np.zeros(n_samples)
    roc_auc_samples = np.zeros(n_samples)
    pr_auc_samples = np.zeros(n_samples)
    
    # Compute bootstrap samples
    n_samples_data = len(y_true)
    
    for i in range(n_samples):
        # Sample with replacement
        indices = np.random.choice(n_samples_data, n_samples_data, replace=True)
        y_true_sample = y_true[indices]
        y_pred_class_sample = y_pred_class[indices]
        y_pred_prob_sample = y_pred_prob[indices]
        
        # Compute metrics
        accuracy_samples[i] = accuracy_score(y_true_sample, y_pred_class_sample)
        precision_samples[i] = precision_score(y_true_sample, y_pred_class_sample)
        recall_samples[i] = recall_score(y_true_sample, y_pred_class_sample)
        f1_samples[i] = f1_score(y_true_sample, y_pred_class_sample)
        
        # Skip ROC AUC and PR AUC if all samples are of the same class
        if len(np.unique(y_true_sample)) > 1:
            roc_auc_samples[i] = roc_auc_score(y_true_sample, y_pred_prob_sample)
            pr_auc_samples[i] = average_precision_score(y_true_sample, y_pred_prob_sample)
        else:
            roc_auc_samples[i] = np.nan
            pr_auc_samples[i] = np.nan
    
    # Compute confidence intervals
    alpha = (1 - confidence) / 2
    
    def compute_ci(samples):
        # Remove NaN values
        samples = samples[~np.isnan(samples)]
        if len(samples) == 0:
            return (np.nan, np.nan)
        lower = np.percentile(samples, 100 * alpha)
        upper = np.percentile(samples, 100 * (1 - alpha))
        return (lower, upper)
    
    metrics_ci = {
        'accuracy': compute_ci(accuracy_samples),
        'precision': compute_ci(precision_samples),
        'recall': compute_ci(recall_samples),
        'f1': compute_ci(f1_samples),
        'roc_auc': compute_ci(roc_auc_samples),
        'pr_auc': compute_ci(pr_auc_samples)
    }
    
    return metrics_ci

def plot_confusion_matrix(cm, class_names, model_type, output_dir='reports/figures'):
    """
    Plot confusion matrix.
    
    Args:
        cm (np.ndarray): Confusion matrix
        class_names (list): Class names
        model_type (str): Model type
        output_dir (str): Output directory
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {model_type.upper()}')
    plt.tight_layout()
    
    # Save figure
    plt.savefig(f"{output_dir}/confusion_matrix_{model_type}.png", dpi=300)
    plt.close()

def plot_roc_curve(fpr, tpr, roc_auc, model_type, output_dir='reports/figures'):
    """
    Plot ROC curve.
    
    Args:
        fpr (np.ndarray): False positive rate
        tpr (np.ndarray): True positive rate
        roc_auc (float): ROC AUC
        model_type (str): Model type
        output_dir (str): Output directory
    """
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_type.upper()}')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(f"{output_dir}/roc_curve_{model_type}.png", dpi=300)
    plt.close()

def plot_pr_curve(precision, recall, pr_auc, model_type, output_dir='reports/figures'):
    """
    Plot precision-recall curve.
    
    Args:
        precision (np.ndarray): Precision
        recall (np.ndarray): Recall
        pr_auc (float): PR AUC
        model_type (str): Model type
        output_dir (str): Output directory
    """
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'PR curve (AUC = {pr_auc:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_type.upper()}')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(f"{output_dir}/pr_curve_{model_type}.png", dpi=300)
    plt.close()

def plot_cascade_timeline(X_test, y_test, y_pred_class, model_type, output_dir='reports/figures'):
    """
    Plot cascade timelines for true positives, false positives, etc.
    
    Args:
        X_test (np.ndarray): Test data
        y_test (np.ndarray): True labels
        y_pred_class (np.ndarray): Predicted classes
        model_type (str): Model type
        output_dir (str): Output directory
    """
    # Flatten arrays
    y_test = y_test.flatten()
    y_pred_class = y_pred_class.flatten()
    
    # Get indices for different prediction categories
    tp_idx = np.where((y_test == 1) & (y_pred_class == 1))[0]
    fp_idx = np.where((y_test == 0) & (y_pred_class == 1))[0]
    tn_idx = np.where((y_test == 0) & (y_pred_class == 0))[0]
    fn_idx = np.where((y_test == 1) & (y_pred_class == 0))[0]
    
    # Sample up to 10 cascades from each category
    n_samples = min(10, min(len(tp_idx), len(fp_idx), len(tn_idx), len(fn_idx)))
    
    if n_samples == 0:
        logger.warning("Not enough samples for cascade timeline visualization")
        return
    
    tp_idx = np.random.choice(tp_idx, n_samples, replace=False)
    fp_idx = np.random.choice(fp_idx, n_samples, replace=False)
    tn_idx = np.random.choice(tn_idx, n_samples, replace=False)
    fn_idx = np.random.choice(fn_idx, n_samples, replace=False)
    
    # Create figure
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot true positives
    for i, idx in enumerate(tp_idx):
        axs[0, 0].plot(np.arange(X_test.shape[1]), X_test[idx], alpha=0.7)
    axs[0, 0].set_title('True Positives')
    axs[0, 0].set_xlabel('Time (hours)')
    axs[0, 0].set_ylabel('Retweets')
    
    # Plot false positives
    for i, idx in enumerate(fp_idx):
        axs[0, 1].plot(np.arange(X_test.shape[1]), X_test[idx], alpha=0.7)
    axs[0, 1].set_title('False Positives')
    axs[0, 1].set_xlabel('Time (hours)')
    axs[0, 1].set_ylabel('Retweets')
    
    # Plot true negatives
    for i, idx in enumerate(tn_idx):
        axs[1, 0].plot(np.arange(X_test.shape[1]), X_test[idx], alpha=0.7)
    axs[1, 0].set_title('True Negatives')
    axs[1, 0].set_xlabel('Time (hours)')
    axs[1, 0].set_ylabel('Retweets')
    
    # Plot false negatives
    for i, idx in enumerate(fn_idx):
        axs[1, 1].plot(np.arange(X_test.shape[1]), X_test[idx], alpha=0.7)
    axs[1, 1].set_title('False Negatives')
    axs[1, 1].set_xlabel('Time (hours)')
    axs[1, 1].set_ylabel('Retweets')
    
    plt.tight_layout()
    plt.suptitle(f'Cascade Timelines - {model_type.upper()}', y=1.02)
    
    # Save figure
    plt.savefig(f"{output_dir}/cascade_timelines_{model_type}.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_dimension_reduction(model, model_type, X_test, y_test, output_dir='reports/figures'):
    """
    Plot dimension reduction visualization (PCA/ICA).
    
    Args:
        model: Model to use
        model_type (str): Model type
        X_test (np.ndarray): Test data
        y_test (np.ndarray): True labels
        output_dir (str): Output directory
    """
    if model_type not in ['pca', 'ica']:
        logger.warning(f"Dimension reduction plot not applicable for {model_type}")
        return
    
    # Flatten labels
    y_test = y_test.flatten()
    
    # Apply dimension reduction
    X_reduced = model.transform(X_test)
    
    # Plot first two components
    plt.figure(figsize=(10, 8))
    
    plt.scatter(X_reduced[y_test == 0, 0], X_reduced[y_test == 0, 1], alpha=0.8, label='Non-viral')
    plt.scatter(X_reduced[y_test == 1, 0], X_reduced[y_test == 1, 1], alpha=0.8, label='Viral')
    
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.title(f'{model_type.upper()} - Top 2 Components')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(f"{output_dir}/{model_type}_components.png", dpi=300)
    plt.close()

def plot_early_detection(model, model_type, X_test, y_test, output_dir='reports/figures'):
    """
    Plot early detection accuracy at different time points.
    
    Args:
        model: Model to use
        model_type (str): Model type
        X_test (np.ndarray): Test data
        y_test (np.ndarray): True labels
        output_dir (str): Output directory
    """
    # Flatten labels
    y_test = y_test.flatten()
    
    # Time points to evaluate (1, 2, 4, 6, 12, 24 hours)
    time_points = [1, 2, 4, 6, 12, 24]
    
    # Initialize arrays for metrics
    accuracy = np.zeros(len(time_points))
    f1 = np.zeros(len(time_points))
    auc = np.zeros(len(time_points))
    
    # Evaluate at each time point
    for i, t in enumerate(time_points):
        if t >= X_test.shape[1]:
            # If time point is beyond data length, use all available data
            X_t = X_test
        else:
            # Create truncated time series
            X_t = X_test.copy()
            X_t[:, t:] = 0
        
        # Generate predictions
        if model_type == 'svm':
            y_pred_class, y_pred_prob = predict(model, model_type, X_t)
        else:
            y_pred_prob = model.predict(X_t)
            y_pred_class = (y_pred_prob > 0.5).astype(int)
            y_pred_class = y_pred_class.flatten()
            y_pred_prob = y_pred_prob.flatten()
        
        # Compute metrics
        accuracy[i] = accuracy_score(y_test, y_pred_class)
        f1[i] = f1_score(y_test, y_pred_class)
        
        # ROC AUC computation might fail if predictions are constant
        try:
            auc[i] = roc_auc_score(y_test, y_pred_prob)
        except:
            auc[i] = np.nan
    
    # Plot metrics
    plt.figure(figsize=(10, 6))
    
    plt.plot(time_points, accuracy, 'o-', label='Accuracy')
    plt.plot(time_points, f1, 's-', label='F1 Score')
    plt.plot(time_points, auc, '^-', label='ROC AUC')
    
    plt.xlabel('Time (hours)')
    plt.ylabel('Metric Value')
    plt.title(f'Early Detection Performance - {model_type.upper()}')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(time_points)
    plt.ylim([0, 1])
    plt.tight_layout()
    
    # Save figure
    plt.savefig(f"{output_dir}/early_detection_{model_type}.png", dpi=300)
    plt.close()

def generate_html_report(model_type, metrics, metrics_ci=None, output_dir='reports'):
    """
    Generate HTML report.
    
    Args:
        model_type (str): Model type
        metrics (dict): Metrics
        metrics_ci (dict, optional): Metrics with confidence intervals
        output_dir (str): Output directory
    """
    # Create HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Evaluation Report - {model_type.upper()}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2 {{ color: #333; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .metrics {{ display: flex; flex-wrap: wrap; }}
            .metric-card {{ background-color: #f9f9f9; border-radius: 10px; padding: 20px; margin: 10px; flex: 1; min-width: 200px; }}
            .metric-value {{ font-size: 24px; font-weight: bold; color: #333; }}
            .metric-ci {{ font-size: 14px; color: #666; }}
            .figures {{ display: flex; flex-wrap: wrap; justify-content: center; }}
            .figure {{ margin: 15px; }}
            .figure img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 5px; }}
            .figure-caption {{ text-align: center; margin-top: 10px; }}
        </style>
    </head>
    <body>
        <h1>Evaluation Report - {model_type.upper()}</h1>
        <p>Generated on {time.strftime("%Y-%m-%d %H:%M:%S")}</p>
        
        <h2>Metrics</h2>
        <div class="metrics">
    """
    
    # Add metrics
    for key in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']:
        if key in metrics:
            html_content += f"""
            <div class="metric-card">
                <h3>{key.replace('_', ' ').title()}</h3>
                <div class="metric-value">{metrics[key]:.3f}</div>
            """
            
            if metrics_ci and key in metrics_ci:
                lower, upper = metrics_ci[key]
                html_content += f"""
                <div class="metric-ci">95% CI: [{lower:.3f}, {upper:.3f}]</div>
                """
            
            html_content += "</div>"
    
    html_content += """
        </div>
        
        <h2>Figures</h2>
        <div class="figures">
    """
    
    # Add figures
    figures = [
        ('confusion_matrix', 'Confusion Matrix'),
        ('roc_curve', 'ROC Curve'),
        ('pr_curve', 'Precision-Recall Curve'),
        ('cascade_timelines', 'Cascade Timelines')
    ]
    
    if model_type in ['pca', 'ica']:
        figures.append((f'{model_type}_components', f'{model_type.upper()} Components'))
    
    figures.append(('early_detection', 'Early Detection Performance'))
    
    for fig_id, fig_caption in figures:
        fig_path = f"figures/{fig_id}_{model_type}.png"
        if os.path.exists(os.path.join(output_dir, fig_path)):
            html_content += f"""
            <div class="figure">
                <img src="{fig_path}" alt="{fig_caption}">
                <div class="figure-caption">{fig_caption}</div>
            </div>
            """
    
    html_content += """
        </div>
    </body>
    </html>
    """
    
    # Write HTML file
    html_path = os.path.join(output_dir, f"evaluation_{model_type}.html")
    with open(html_path, 'w') as f:
        f.write(html_content)
    
    logger.info(f"HTML report generated at {html_path}")

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set random seed
    np.random.seed(args.random_seed)
    
    # Load data
    _, _, _, _, X_test, y_test = load_data(config)
    
    # Initialize and load model
    model = get_model(args.model, config, args.ckpt)
    
    # Generate predictions
    y_pred_class, y_pred_prob = predict(model, args.model, X_test)
    
    # Compute metrics
    metrics = compute_metrics(y_test, y_pred_class, y_pred_prob)
    
    # Print metrics
    logger.info(f"Evaluation metrics for {args.model}:")
    for key, value in metrics.items():
        if key not in ['confusion_matrix', 'roc_curve', 'pr_curve']:
            logger.info(f"{key.replace('_', ' ').title()}: {value:.4f}")
    
    # Compute bootstrap confidence intervals if requested
    metrics_ci = None
    if args.bootstrap:
        logger.info(f"Computing bootstrap confidence intervals ({args.bootstrap_samples} samples)...")
        metrics_ci = bootstrap_metrics(y_test, y_pred_class, y_pred_prob, n_samples=args.bootstrap_samples)
        
        logger.info("Bootstrap confidence intervals:")
        for key, (lower, upper) in metrics_ci.items():
            logger.info(f"{key.replace('_', ' ').title()}: [{lower:.4f}, {upper:.4f}]")
    
    # Generate plots
    logger.info("Generating plots...")
    
    # Confusion matrix
    plot_confusion_matrix(
        metrics['confusion_matrix'],
        class_names=['Non-viral', 'Viral'],
        model_type=args.model
    )
    
    # ROC curve
    plot_roc_curve(
        *metrics['roc_curve'],
        metrics['roc_auc'],
        model_type=args.model
    )
    
    # PR curve
    plot_pr_curve(
        *metrics['pr_curve'],
        metrics['pr_auc'],
        model_type=args.model
    )
    
    # Cascade timeline visualization
    plot_cascade_timeline(X_test, y_test, y_pred_class, args.model)
    
    # Dimension reduction visualization for PCA/ICA
    if args.model in ['pca', 'ica']:
        plot_dimension_reduction(model, args.model, X_test, y_test)
    
    # Early detection performance
    plot_early_detection(model, args.model, X_test, y_test)
    
    # Generate HTML report
    generate_html_report(args.model, metrics, metrics_ci)
    
    logger.info("Evaluation completed")

if __name__ == "__main__":
    main() 