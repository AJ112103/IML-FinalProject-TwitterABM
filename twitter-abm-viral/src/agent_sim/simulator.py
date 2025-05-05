#!/usr/bin/env python3
"""
Twitter retweet cascade simulator using agent-based modeling.
Adapted to work with Twitter retweet analysis dataset.
"""

import os
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from scipy.optimize import minimize
from scipy.stats import wasserstein_distance

from .retweet_agent import RetweetAgent

logger = logging.getLogger("simulator")

class RetweetSimulator:
    """
    Agent-based simulator for Twitter retweet cascades.
    
    The simulator creates a network of agents (Twitter users) and simulates
    how tweets propagate through the network via retweets.
    """
    
    def __init__(self, config, random_seed=None):
        """
        Initialize the simulator with given configuration.
        
        Args:
            config (dict): Configuration dictionary
            random_seed (int, optional): Random seed for reproducibility
        """
        self.config = config
        
        # Set random seed for reproducibility
        if random_seed is not None:
            np.random.seed(random_seed)
        elif 'random_seed' in config:
            np.random.seed(config['random_seed'])
            
        # Initialize simulator parameters
        self.num_agents = config['abm']['num_agents']
        self.simulation_steps = config['abm']['simulation_steps']
        self.influence_factor = config['abm']['influence_factor']
        self.susceptibility_factor = config['abm']['susceptibility_factor']
        self.network_density = config['abm']['network_density']
        
        # Initialize agents and network
        self.agents = []
        self.follower_network = None
        self.initialized = False
        
        logger.info(f"Initialized simulator with {self.num_agents} agents")
        
    def initialize_network(self, user_data=None):
        """
        Initialize the network of agents and their follower relationships.
        
        Args:
            user_data (pd.DataFrame, optional): User data from processed dataset
        """
        logger.info("Initializing agent network")
        
        # Create agents
        self.agents = []
        
        if user_data is not None and not user_data.empty:
            # Use actual user data to initialize agents
            logger.info("Using real user data to initialize agents")
            
            # Subsample if we have more users than agents
            if len(user_data) > self.num_agents:
                user_data = user_data.sample(n=self.num_agents, random_state=self.config['random_seed'])
            
            # Fill remaining agents with synthetic data if needed
            if len(user_data) < self.num_agents:
                num_synthetic = self.num_agents - len(user_data)
                logger.info(f"Adding {num_synthetic} synthetic agents to reach target count")
            
            # Create agents from user data
            for _, user in user_data.iterrows():
                agent = RetweetAgent(
                    followers=user['followers'],
                    influence=user['influence'],
                    susceptibility=user['susceptibility'],
                    sentiment_bias=user['sentiment'] / 5.0  # Normalize to [-1, 1]
                )
                self.agents.append(agent)
            
            # Add synthetic agents if needed
            while len(self.agents) < self.num_agents:
                # Create synthetic agent with parameters sampled from user distributions
                followers = np.random.lognormal(
                    np.log(user_data['followers'].mean()), 
                    np.log(user_data['followers'].std())
                )
                influence = np.clip(np.random.normal(
                    user_data['influence'].mean(),
                    user_data['influence'].std()
                ), 0, 1)
                susceptibility = np.clip(np.random.normal(
                    user_data['susceptibility'].mean(),
                    user_data['susceptibility'].std()
                ), 0, 1)
                sentiment_bias = np.clip(np.random.normal(0, 0.3), -1, 1)
                
                agent = RetweetAgent(
                    followers=int(followers),
                    influence=influence,
                    susceptibility=susceptibility,
                    sentiment_bias=sentiment_bias
                )
                self.agents.append(agent)
                
        else:
            # Create synthetic agents
            logger.info("Creating synthetic agents")
            
            # Followers follow power law distribution (common in social networks)
            followers_scale = 500  # Scale parameter for lognormal distribution
            
            for i in range(self.num_agents):
                # Follower count from lognormal distribution
                followers = int(np.random.lognormal(mean=np.log(followers_scale), sigma=1.5))
                
                # Influence positively correlated with follower count but with variation
                influence = np.clip(0.1 + 0.5 * (np.log1p(followers) / np.log1p(100000)) + np.random.normal(0, 0.1), 0, 1)
                
                # Susceptibility varies independently
                susceptibility = np.clip(np.random.beta(2, 3), 0, 1)
                
                # Sentiment bias (how much emotional content affects this agent)
                sentiment_bias = np.clip(np.random.normal(0, 0.3), -1, 1)
                
                agent = RetweetAgent(
                    followers=followers,
                    influence=influence,
                    susceptibility=susceptibility,
                    sentiment_bias=sentiment_bias
                )
                self.agents.append(agent)
        
        # Build follower network (who follows whom)
        logger.info("Building follower network")
        self.follower_network = np.zeros((self.num_agents, self.num_agents), dtype=bool)
        
        # Preferential attachment: agents with more followers are more likely to be followed
        follower_counts = np.array([agent.followers for agent in self.agents])
        follow_probabilities = follower_counts / follower_counts.sum()
        
        # Each agent follows others based on network density and preferential attachment
        avg_connections = int(self.num_agents * self.network_density)
        for i in range(self.num_agents):
            # How many accounts this agent follows (proportional to their activity)
            num_follows = max(5, int(np.random.poisson(avg_connections * self.agents[i].susceptibility)))
            
            # Who they follow (weighted by popularity)
            follows = np.random.choice(
                self.num_agents, 
                size=min(num_follows, self.num_agents-1), 
                replace=False, 
                p=follow_probabilities
            )
            
            # Remove self-follows
            follows = follows[follows != i]
            
            for j in follows:
                self.follower_network[i, j] = True
        
        total_connections = self.follower_network.sum()
        logger.info(f"Follower network created with {total_connections} connections " +
                    f"(density: {total_connections / (self.num_agents**2 - self.num_agents):.4f})")
        
        self.initialized = True
        
    def reset_simulation(self):
        """Reset the simulation state of all agents while keeping the network intact."""
        if not self.initialized:
            raise RuntimeError("Simulator not initialized. Call initialize_network() first.")
        
        logger.debug("Resetting simulation")
        for agent in self.agents:
            agent.reset()
    
    def simulate_cascade(self, initial_seed_fraction=0.01, sentiment=0.0, tweet_quality=0.5):
        """
        Simulate a retweet cascade starting from seed users.
        
        Args:
            initial_seed_fraction (float): Fraction of users who see the tweet initially
            sentiment (float): Sentiment score of the tweet (-1 to 1)
            tweet_quality (float): Intrinsic quality/appeal of the tweet (0 to 1)
            
        Returns:
            tuple: (retweet_counts, time_series, cascade_size, is_viral)
        """
        if not self.initialized:
            raise RuntimeError("Simulator not initialized. Call initialize_network() first.")
        
        self.reset_simulation()
        
        # Select initial seed users (weighted by follower count)
        followers = np.array([agent.followers for agent in self.agents])
        seed_probabilities = followers / followers.sum()
        
        num_seeds = max(1, int(self.num_agents * initial_seed_fraction))
        seed_indices = np.random.choice(
            self.num_agents,
            size=min(num_seeds, self.num_agents),
            replace=False,
            p=seed_probabilities
        )
        
        # Initial tweet exposure
        for idx in seed_indices:
            self.agents[idx].seen = True
            self.agents[idx].time_seen = 0
        
        # Simulate the cascading retweet process over time
        retweet_time_series = np.zeros(self.simulation_steps)
        
        for step in range(self.simulation_steps):
            # Process agents who have seen the tweet but haven't yet decided
            newly_retweeted = []
            
            for i, agent in enumerate(self.agents):
                # Skip if agent hasn't seen tweet or has already made a decision
                if not agent.seen or agent.retweeted is not None:
                    continue
                
                # Determine if agent will retweet based on various factors
                # - Time since seeing the tweet (decay with time)
                time_factor = np.exp(-0.3 * (step - agent.time_seen))  # Increased decay
                
                # - Influence of original tweet's author (simplified in this model)
                influence_effect = self.influence_factor * 0.5  # Reduced overall influence
                
                # - Intrinsic tweet quality
                quality_effect = tweet_quality * 0.7  # Reduce quality impact
                
                # - Sentiment alignment
                sentiment_effect = 1.0 + agent.sentiment_bias * sentiment * 0.5  # Reduced sentiment impact
                
                # Combine factors for retweet probability
                retweet_prob = agent.susceptibility * influence_effect * quality_effect * sentiment_effect * time_factor
                retweet_prob = min(retweet_prob * 0.4, 0.8)  # Strongly reduce maximum probability
                
                # Decide whether to retweet
                if np.random.random() < retweet_prob:
                    agent.retweeted = True
                    agent.time_retweeted = step
                    newly_retweeted.append(i)
                else:
                    # Once decided not to retweet, won't reconsider
                    agent.retweeted = False
            
            # Record retweets for this time step
            retweet_time_series[step] = len(newly_retweeted)
            
            # Propagate to followers
            for retweeter_idx in newly_retweeted:
                # Find all followers of this retweeter
                followers = np.where(self.follower_network[:, retweeter_idx])[0]
                
                for follower_idx in followers:
                    # If follower hasn't seen the tweet yet
                    if not self.agents[follower_idx].seen:
                        self.agents[follower_idx].seen = True
                        self.agents[follower_idx].time_seen = step
        
        # Calculate total retweet count
        total_retweets = sum(1 for agent in self.agents if agent.retweeted)
        
        # Determine if viral based on our 90th percentile threshold (16 retweets)
        is_viral = total_retweets > 16
        
        # For multi-class:
        virality_class = 0  # Not viral
        if total_retweets > 5:
            virality_class = 1  # Moderately viral
        if total_retweets > 20:
            virality_class = 2  # Highly viral
        
        return retweet_time_series, total_retweets, is_viral, virality_class
    
    def calibrate(self, target_series, max_iter=100):
        """
        Calibrate the simulator parameters to match observed retweet patterns.
        
        Args:
            target_series (np.ndarray): Target retweet time series to match
            max_iter (int): Maximum optimization iterations
            
        Returns:
            dict: Optimized parameters
        """
        logger.info("Calibrating simulator parameters")
        
        if not self.initialized:
            self.initialize_network()
        
        # Normalize target series
        target_series = target_series / np.sum(target_series)
        
        # Define parameter bounds
        bounds = [
            (0.001, 0.1),    # initial_seed_fraction
            (-1.0, 1.0),     # sentiment
            (0.1, 0.9),      # tweet_quality
            (0.1, 0.9),      # influence_factor
            (0.1, 0.9)       # susceptibility_factor
        ]
        
        # Initial parameter guess
        initial_params = [0.01, 0.0, 0.5, self.influence_factor, self.susceptibility_factor]
        
        # Define objective function to minimize (Earth Mover's Distance)
        def objective(params):
            seed_frac, sentiment, quality, infl_factor, susc_factor = params
            
            # Update model parameters
            old_infl = self.influence_factor
            old_susc = self.susceptibility_factor
            self.influence_factor = infl_factor
            self.susceptibility_factor = susc_factor
            
            # Run simulation
            try:
                sim_series, _, _, _ = self.simulate_cascade(
                    initial_seed_fraction=seed_frac,
                    sentiment=sentiment,
                    tweet_quality=quality
                )
                
                # Normalize series
                if np.sum(sim_series) > 0:
                    sim_series = sim_series / np.sum(sim_series)
                
                # Calculate Earth Mover's Distance (Wasserstein distance)
                distance = wasserstein_distance(target_series, sim_series)
                return distance
            except Exception as e:
                logger.error(f"Error in calibration: {e}")
                return float('inf')
            finally:
                # Restore original parameters
                self.influence_factor = old_infl
                self.susceptibility_factor = old_susc
        
        # Run optimization
        try:
            result = minimize(
                objective,
                initial_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iter}
            )
            
            # Update model with optimized parameters
            seed_frac, sentiment, quality, infl_factor, susc_factor = result.x
            self.influence_factor = infl_factor
            self.susceptibility_factor = susc_factor
            
            logger.info(f"Calibration complete. Optimized parameters: seed={seed_frac:.4f}, " +
                        f"sentiment={sentiment:.4f}, quality={quality:.4f}, " +
                        f"influence={infl_factor:.4f}, susceptibility={susc_factor:.4f}")
            
            # Return optimized parameters
            return {
                'seed_fraction': seed_frac,
                'sentiment': sentiment,
                'tweet_quality': quality,
                'influence_factor': infl_factor,
                'susceptibility_factor': susc_factor,
                'optimization_success': result.success,
                'distance': result.fun
            }
            
        except Exception as e:
            logger.error(f"Calibration failed: {e}")
            return {
                'optimization_success': False,
                'error': str(e)
            }
    
    def generate_training_data(self, num_samples, output_path=None):
        """
        Generate training data for ML models by simulating multiple cascades.
        
        Args:
            num_samples (int): Number of simulated cascades to generate
            output_path (str, optional): Path to save the generated data
            
        Returns:
            pd.DataFrame: Generated training data
        """
        if not self.initialized:
            self.initialize_network()
        
        logger.info(f"Generating {num_samples} simulated cascades for training data")
        
        data = []
        for i in tqdm(range(num_samples), desc="Simulating cascades"):
            # Randomize parameters for variety
            seed_fraction = np.random.uniform(0.001, 0.05)
            sentiment = np.random.uniform(-1, 1)
            tweet_quality = np.random.uniform(0.2, 0.8)
            
            # Run simulation
            time_series, total_retweets, is_viral, virality_class = self.simulate_cascade(
                initial_seed_fraction=seed_fraction,
                sentiment=sentiment,
                tweet_quality=tweet_quality
            )
            
            # Extract features from time series
            early_retweets = time_series[:12].sum()  # First 12 steps
            peak_time = np.argmax(time_series)
            peak_volume = np.max(time_series)
            
            # Store simulation results
            data.append({
                'time_series': time_series.tolist(),
                'total_retweets': total_retweets,
                'is_viral': int(is_viral),
                'virality_class': virality_class,
                'sentiment': sentiment,
                'tweet_quality': tweet_quality,
                'seed_fraction': seed_fraction,
                'early_retweets': early_retweets,
                'peak_time': peak_time,
                'peak_volume': peak_volume
            })
        
        # Create dataframe
        df = pd.DataFrame(data)
        
        # Save if output path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            df.to_csv(output_path, index=False)
            logger.info(f"Saved {len(df)} simulated cascades to {output_path}")
        
        return df

    def visualize_cascade(self, time_series, title=None):
        """
        Visualize a retweet cascade time series.
        
        Args:
            time_series (np.ndarray): Retweet time series
            title (str, optional): Plot title
        """
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(10, 6))
            plt.plot(time_series, 'b-', linewidth=2)
            plt.fill_between(range(len(time_series)), time_series, alpha=0.3)
            plt.xlabel('Time Steps')
            plt.ylabel('Number of Retweets')
            plt.title(title or 'Retweet Cascade Simulation')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            return plt
            
        except ImportError:
            logger.warning("Matplotlib not available. Unable to visualize cascade.")
            return None 