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
    
    def __init__(self, config, random_seed=None):

        self.config = config

        if random_seed is not None:
            np.random.seed(random_seed)
        elif 'random_seed' in config:
            np.random.seed(config['random_seed'])

        self.num_agents = config['abm']['num_agents']
        self.simulation_steps = config['abm']['simulation_steps']
        self.influence_factor = config['abm']['influence_factor']
        self.susceptibility_factor = config['abm']['susceptibility_factor']
        self.network_density = config['abm']['network_density']

        self.agents = []
        self.follower_network = None
        self.initialized = False
        
        logger.info(f"Initialized simulator with {self.num_agents} agents")
        
    def initialize_network(self, user_data=None):
        
        logger.info("Initializing agent network")

        self.agents = []
        
        if user_data is not None and not user_data.empty:
            logger.info("Using real user data to initialize agents")

            if len(user_data) > self.num_agents:
                user_data = user_data.sample(n=self.num_agents, random_state=self.config['random_seed'])

            if len(user_data) < self.num_agents:
                num_synthetic = self.num_agents - len(user_data)
                logger.info(f"Adding {num_synthetic} synthetic agents to reach target count")

            for _, user in user_data.iterrows():
                agent = RetweetAgent(
                    followers=user['followers'],
                    influence=user['influence'],
                    susceptibility=user['susceptibility'],
                    sentiment_bias=user['sentiment'] / 5.0  # Normalize to [-1, 1]
                )
                self.agents.append(agent)

            while len(self.agents) < self.num_agents:
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
            logger.info("Creating synthetic agents")

            followers_scale = 500 
            
            for i in range(self.num_agents):
                followers = int(np.random.lognormal(mean=np.log(followers_scale), sigma=1.5))

                influence = np.clip(0.1 + 0.5 * (np.log1p(followers) / np.log1p(100000)) + np.random.normal(0, 0.1), 0, 1)

                susceptibility = np.clip(np.random.beta(2, 3), 0, 1)

                sentiment_bias = np.clip(np.random.normal(0, 0.3), -1, 1)
                
                agent = RetweetAgent(
                    followers=followers,
                    influence=influence,
                    susceptibility=susceptibility,
                    sentiment_bias=sentiment_bias
                )
                self.agents.append(agent)

        logger.info("Building follower network")
        self.follower_network = np.zeros((self.num_agents, self.num_agents), dtype=bool)

        follower_counts = np.array([agent.followers for agent in self.agents])
        follow_probabilities = follower_counts / follower_counts.sum()

        avg_connections = int(self.num_agents * self.network_density)
        for i in range(self.num_agents):
            num_follows = max(5, int(np.random.poisson(avg_connections * self.agents[i].susceptibility)))

            follows = np.random.choice(
                self.num_agents, 
                size=min(num_follows, self.num_agents-1), 
                replace=False, 
                p=follow_probabilities
            )

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
        if not self.initialized:
            raise RuntimeError("Simulator not initialized. Call initialize_network() first.")
        
        self.reset_simulation()

        followers = np.array([agent.followers for agent in self.agents])
        seed_probabilities = followers / followers.sum()
        
        num_seeds = max(1, int(self.num_agents * initial_seed_fraction))
        seed_indices = np.random.choice(
            self.num_agents,
            size=min(num_seeds, self.num_agents),
            replace=False,
            p=seed_probabilities
        )

        for idx in seed_indices:
            self.agents[idx].seen = True
            self.agents[idx].time_seen = 0

        retweet_time_series = np.zeros(self.simulation_steps)
        
        for step in range(self.simulation_steps):
            newly_retweeted = []
            
            for i, agent in enumerate(self.agents):
                if not agent.seen or agent.retweeted is not None:
                    continue

                time_factor = np.exp(-0.3 * (step - agent.time_seen))

                influence_effect = self.influence_factor * 0.5

                quality_effect = tweet_quality * 0.7

                sentiment_effect = 1.0 + agent.sentiment_bias * sentiment * 0.5

                retweet_prob = agent.susceptibility * influence_effect * quality_effect * sentiment_effect * time_factor
                retweet_prob = min(retweet_prob * 0.4, 0.8)

                if np.random.random() < retweet_prob:
                    agent.retweeted = True
                    agent.time_retweeted = step
                    newly_retweeted.append(i)
                else:
                    agent.retweeted = False

            retweet_time_series[step] = len(newly_retweeted)

            for retweeter_idx in newly_retweeted:
                followers = np.where(self.follower_network[:, retweeter_idx])[0]
                
                for follower_idx in followers:
                    if not self.agents[follower_idx].seen:
                        self.agents[follower_idx].seen = True
                        self.agents[follower_idx].time_seen = step

        total_retweets = sum(1 for agent in self.agents if agent.retweeted)

        is_viral = total_retweets > 16

        virality_class = 0
        if total_retweets > 5:
            virality_class = 1
        if total_retweets > 20:
            virality_class = 2
        
        return retweet_time_series, total_retweets, is_viral, virality_class
    
    def calibrate(self, target_series, max_iter=100):

        logger.info("Calibrating simulator parameters")
        
        if not self.initialized:
            self.initialize_network()

        target_series = target_series / np.sum(target_series)

        bounds = [
            (0.001, 0.1),   
            (-1.0, 1.0),     
            (0.1, 0.9),      
            (0.1, 0.9),     
            (0.1, 0.9)       
        ]

        initial_params = [0.01, 0.0, 0.5, self.influence_factor, self.susceptibility_factor]

        def objective(params):
            seed_frac, sentiment, quality, infl_factor, susc_factor = params

            old_infl = self.influence_factor
            old_susc = self.susceptibility_factor
            self.influence_factor = infl_factor
            self.susceptibility_factor = susc_factor

            try:
                sim_series, _, _, _ = self.simulate_cascade(
                    initial_seed_fraction=seed_frac,
                    sentiment=sentiment,
                    tweet_quality=quality
                )

                if np.sum(sim_series) > 0:
                    sim_series = sim_series / np.sum(sim_series)

                distance = wasserstein_distance(target_series, sim_series)
                return distance
            except Exception as e:
                logger.error(f"Error in calibration: {e}")
                return float('inf')
            finally:
                self.influence_factor = old_infl
                self.susceptibility_factor = old_susc

        try:
            result = minimize(
                objective,
                initial_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iter}
            )

            seed_frac, sentiment, quality, infl_factor, susc_factor = result.x
            self.influence_factor = infl_factor
            self.susceptibility_factor = susc_factor
            
            logger.info(f"Calibration complete. Optimized parameters: seed={seed_frac:.4f}, " +
                        f"sentiment={sentiment:.4f}, quality={quality:.4f}, " +
                        f"influence={infl_factor:.4f}, susceptibility={susc_factor:.4f}")

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

        if not self.initialized:
            self.initialize_network()
        
        logger.info(f"Generating {num_samples} simulated cascades for training data")
        
        data = []
        for i in tqdm(range(num_samples), desc="Simulating cascades"):
            seed_fraction = np.random.uniform(0.001, 0.05)
            sentiment = np.random.uniform(-1, 1)
            tweet_quality = np.random.uniform(0.2, 0.8)

            time_series, total_retweets, is_viral, virality_class = self.simulate_cascade(
                initial_seed_fraction=seed_fraction,
                sentiment=sentiment,
                tweet_quality=tweet_quality
            )

            early_retweets = time_series[:12].sum()  # First 12 steps
            peak_time = np.argmax(time_series)
            peak_volume = np.max(time_series)

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

        df = pd.DataFrame(data)

        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            df.to_csv(output_path, index=False)
            logger.info(f"Saved {len(df)} simulated cascades to {output_path}")
        
        return df

    def visualize_cascade(self, time_series, title=None):

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