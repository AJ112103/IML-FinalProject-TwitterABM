#!/usr/bin/env python3
"""
Test script for the RetweetSimulator with our processed data.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import logging
from src.agent_sim.simulator import RetweetSimulator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('test_simulator')

def main():
    # Load config
    with open('configs/default.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load processed user data for network initialization
    user_data_path = os.path.join(config['data']['processed_dir'], 'user_network.csv')
    logger.info(f"Loading user network data from {user_data_path}")
    user_data = pd.read_csv(user_data_path)
    
    # Initialize simulator
    logger.info("Initializing simulator")
    simulator = RetweetSimulator(config, random_seed=config['random_seed'])
    
    # Initialize network with real user data
    logger.info("Building network with real user data")
    simulator.initialize_network(user_data=user_data)
    
    # Test case 1: Basic simulation
    logger.info("Test Case 1: Running basic simulation")
    time_series, total_retweets, is_viral, virality_class = simulator.simulate_cascade(
        initial_seed_fraction=0.01,
        sentiment=0.2,
        tweet_quality=0.6
    )
    
    print(f"Simulation results: {total_retweets} total retweets")
    print(f"Is viral: {is_viral}")
    print(f"Virality class: {virality_class}")
    
    # Plot time series
    plt.figure(figsize=(10, 5))
    plt.plot(time_series, 'b-', linewidth=2)
    plt.fill_between(range(len(time_series)), time_series, alpha=0.3)
    plt.title(f"Simulated Retweet Cascade (Total: {total_retweets}, Viral: {is_viral})")
    plt.xlabel("Time Steps")
    plt.ylabel("Number of Retweets")
    plt.grid(True, alpha=0.3)
    
    # Create output directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/basic_simulation.png')
    plt.close()
    
    # Test case 2: Generate multiple cascades with different parameters
    logger.info("Test Case 2: Generating multiple cascades")
    
    # Parameters to test
    seeds = [0.01, 0.02, 0.05]
    sentiments = [-0.5, 0.0, 0.5]
    qualities = [0.3, 0.5, 0.7]
    
    results = []
    
    for seed in seeds:
        for sentiment in sentiments:
            for quality in qualities:
                time_series, total_retweets, is_viral, virality_class = simulator.simulate_cascade(
                    initial_seed_fraction=seed,
                    sentiment=sentiment,
                    tweet_quality=quality
                )
                
                results.append({
                    'seed_fraction': seed,
                    'sentiment': sentiment,
                    'tweet_quality': quality,
                    'total_retweets': total_retweets,
                    'is_viral': is_viral,
                    'virality_class': virality_class
                })
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    results_df.to_csv('results/simulation_results.csv', index=False)
    
    # Calculate viral percentage
    viral_pct = results_df['is_viral'].mean() * 100
    print(f"\nViral percentage across all simulations: {viral_pct:.2f}%")
    
    # Print effect of each parameter on virality
    print("\nEffect of parameters on virality:")
    
    print("\nSeed Fraction:")
    for seed in seeds:
        subset = results_df[results_df['seed_fraction'] == seed]
        viral_pct = subset['is_viral'].mean() * 100
        print(f"  {seed:.2f}: {viral_pct:.2f}% viral")
    
    print("\nSentiment:")
    for sentiment in sentiments:
        subset = results_df[results_df['sentiment'] == sentiment]
        viral_pct = subset['is_viral'].mean() * 100
        print(f"  {sentiment:.1f}: {viral_pct:.2f}% viral")
    
    print("\nTweet Quality:")
    for quality in qualities:
        subset = results_df[results_df['tweet_quality'] == quality]
        viral_pct = subset['is_viral'].mean() * 100
        print(f"  {quality:.1f}: {viral_pct:.2f}% viral")
    
    # Test case 3: Generate training data
    logger.info("Test Case 3: Generating training data")
    train_data = simulator.generate_training_data(
        num_samples=50,
        output_path='results/training_data.csv'
    )
    
    print(f"\nGenerated {len(train_data)} training samples")
    print(f"Viral percentage in training data: {train_data['is_viral'].mean() * 100:.2f}%")
    
    logger.info("Simulator tests completed successfully")

if __name__ == "__main__":
    main() 