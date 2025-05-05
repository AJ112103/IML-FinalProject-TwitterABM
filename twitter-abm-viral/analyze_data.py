#!/usr/bin/env python3
"""
Analyze the Twitter retweet dataset to identify good metrics for virality
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load the dataset
df = pd.read_csv('data/raw/retweet_analysis.csv')

print("Dataset shape:", df.shape)
print("\nBasic statistics:")
print(df.describe())

# Analyze retweet count distribution
print("\n--- Retweet Count Analysis ---")
print("Percentiles for retweetcount:")
for p in [50, 75, 80, 85, 90, 95, 99]:
    print(f"{p}th percentile: {df.retweetcount.quantile(p/100)}")

print("\nPercentage of tweets with:")
print(f"retweetcount > 0: {(df.retweetcount > 0).mean()*100:.2f}%")
print(f"retweetcount > 5: {(df.retweetcount > 5).mean()*100:.2f}%")
print(f"retweetcount > 10: {(df.retweetcount > 10).mean()*100:.2f}%")
print(f"retweetcount > 20: {(df.retweetcount > 20).mean()*100:.2f}%")

# Analyze reach
print("\n--- Reach Analysis ---")
print("Percentiles for reach:")
for p in [50, 75, 90, 95, 99]:
    print(f"{p}th percentile: {df.reach.quantile(p/100)}")

# Analyze correlation between features
print("\n--- Correlation Analysis ---")
corr = df[['reach', 'retweetcount', 'likes', 'klout', 'sentiment']].corr()
print(corr)

# Analyze viral vs non-viral based on different thresholds
print("\n--- Potential Virality Thresholds ---")

# Based on retweetcount
rt_threshold = df.retweetcount.quantile(0.9)  # 90th percentile
print(f"Retweet count threshold (90th percentile): {rt_threshold}")
df['viral_by_rt'] = df.retweetcount > rt_threshold

# Based on reach
reach_threshold = df.reach.quantile(0.9)  # 90th percentile
print(f"Reach threshold (90th percentile): {reach_threshold}")
df['viral_by_reach'] = df.reach > reach_threshold

# Based on composite score (weighted combination of metrics)
# Normalize each metric to 0-1 range
df['rt_norm'] = df.retweetcount / df.retweetcount.max()
df['reach_norm'] = df.reach / df.reach.max()
df['likes_norm'] = df.likes / df.likes.max()

# Create a composite score
df['virality_score'] = (0.5 * df.rt_norm + 
                         0.3 * df.reach_norm + 
                         0.2 * df.likes_norm)

score_threshold = df.virality_score.quantile(0.9)
print(f"Virality score threshold (90th percentile): {score_threshold}")
df['viral_by_score'] = df.virality_score > score_threshold

# Compare different virality definitions
print("\n--- Comparison of Virality Definitions ---")
print(f"Tweets classified as viral by retweet count: {df.viral_by_rt.sum()} ({df.viral_by_rt.mean()*100:.2f}%)")
print(f"Tweets classified as viral by reach: {df.viral_by_reach.sum()} ({df.viral_by_reach.mean()*100:.2f}%)")
print(f"Tweets classified as viral by composite score: {df.viral_by_score.sum()} ({df.viral_by_score.mean()*100:.2f}%)")

# Overlap between definitions
rt_and_reach = df[df.viral_by_rt & df.viral_by_reach].shape[0]
rt_and_score = df[df.viral_by_rt & df.viral_by_score].shape[0]
reach_and_score = df[df.viral_by_reach & df.viral_by_score].shape[0]
all_three = df[df.viral_by_rt & df.viral_by_reach & df.viral_by_score].shape[0]

print(f"\nOverlap between RT and reach definitions: {rt_and_reach} tweets ({rt_and_reach/df.viral_by_rt.sum()*100:.2f}% of RT-viral)")
print(f"Overlap between RT and score definitions: {rt_and_score} tweets ({rt_and_score/df.viral_by_rt.sum()*100:.2f}% of RT-viral)")
print(f"Overlap between reach and score definitions: {reach_and_score} tweets ({reach_and_score/df.viral_by_reach.sum()*100:.2f}% of reach-viral)")
print(f"Tweets classified as viral by all three definitions: {all_three}")

print("\n--- Recommendation ---")
print("Based on the analysis, recommended virality metric:")
print(f"1. Retweet count > {rt_threshold} (90th percentile)")
print(f"2. Composite score > {score_threshold:.4f} (weighted RT, reach, likes)")
print("3. For more nuanced analysis, use multi-class approach:")
print("   - Not viral: retweetcount <= 5")
print("   - Moderately viral: 5 < retweetcount <= 20")
print("   - Highly viral: retweetcount > 20") 