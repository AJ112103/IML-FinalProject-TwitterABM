#!/usr/bin/env python3
"""
RetweetAgent class for the Twitter Agent-Based Model.
Adapted to work with Twitter retweet analysis dataset.
"""

import numpy as np

class RetweetAgent:
    """
    Agent representing a Twitter user in the retweet cascade simulation.
    
    Attributes:
        followers (int): Number of followers this agent has
        influence (float): Klout score or influence rating (0-1)
        susceptibility (float): How likely the agent is to retweet content (0-1)
        sentiment_bias (float): How much sentiment affects their retweet behavior (-1 to 1)
        seen (bool): Whether the agent has seen the tweet
        retweeted (bool): Whether the agent has retweeted the tweet
        time_seen (int): When the agent saw the tweet
        time_retweeted (int): When the agent retweeted the tweet
    """
    
    def __init__(self, followers=100, influence=None, susceptibility=None, sentiment_bias=None):
        """
        Initialize a retweet agent.
        
        Args:
            followers (int): Number of followers
            influence (float, optional): User's influence score (0-1)
            susceptibility (float, optional): Tendency to retweet (0-1)
            sentiment_bias (float, optional): Sentiment bias (-1 to 1)
        """
        # Basic properties
        self.followers = max(1, int(followers))
        
        # Set influence (defaults to 0.1-0.7 correlated with follower count)
        if influence is not None:
            self.influence = np.clip(influence, 0, 1)
        else:
            # Default: scale with log of followers
            base_influence = 0.1 + 0.4 * np.log1p(self.followers) / np.log1p(10000)
            self.influence = np.clip(base_influence, 0.1, 0.9)
        
        # Set susceptibility to retweet (defaults to beta distribution)
        if susceptibility is not None:
            self.susceptibility = np.clip(susceptibility, 0, 1)
        else:
            # Default: random beta distribution (usually between 0.1 and 0.7)
            self.susceptibility = np.clip(np.random.beta(2, 3), 0, 1)
            
        # Set sentiment bias (how much sentiment affects this agent)
        if sentiment_bias is not None:
            self.sentiment_bias = np.clip(sentiment_bias, -1, 1)
        else:
            # Default: normal distribution centered around 0
            self.sentiment_bias = np.clip(np.random.normal(0, 0.3), -1, 1)
        
        # State variables
        self.seen = False             # Has agent seen the tweet?
        self.retweeted = None         # Has agent retweeted? (None=undecided, True=yes, False=no)
        self.time_seen = None         # When agent saw the tweet
        self.time_retweeted = None    # When agent retweeted
    
    def reset(self):
        """Reset the agent's state for a new simulation."""
        self.seen = False
        self.retweeted = None
        self.time_seen = None
        self.time_retweeted = None
    
    def see_tweet(self, time_step):
        """
        Agent sees the tweet.
        
        Args:
            time_step (int): Current time step in the simulation
        
        Returns:
            bool: True if this is the first time seeing the tweet, False if already seen
        """
        if not self.seen:
            self.seen = True
            self.time_seen = time_step
            return True
        return False
    
    def decide_retweet(self, tweet_quality, tweet_sentiment, time_step):
        """
        Agent decides whether to retweet.
        
        Args:
            tweet_quality (float): Quality/appeal of the tweet (0-1)
            tweet_sentiment (float): Sentiment of the tweet (-5 to 5 scale)
            time_step (int): Current time step in the simulation
        
        Returns:
            bool: True if agent decides to retweet, False otherwise
        """
        if not self.seen or self.retweeted is not None:
            return False
        
        # Calculate time decay factor (less likely to retweet older tweets)
        time_elapsed = time_step - self.time_seen
        time_decay = np.exp(-0.1 * time_elapsed)  # Exponential decay
        
        # Calculate influence factor (higher influence means more likely to be retweeted)
        # Scale influence to 0-1 range
        influence_factor = self.influence
        
        # Calculate sentiment effect (how much sentiment affects this user's decision)
        # Normalize tweet sentiment to -1 to 1 range for calculation
        normalized_sentiment = tweet_sentiment / 5
        
        # Positive sentiment_bias means user prefers positive tweets
        # Negative sentiment_bias means user prefers negative tweets
        sentiment_alignment = 1 - abs(normalized_sentiment - self.sentiment_bias)
        
        # Calculate retweet probability
        # Factors:
        # - Agent's susceptibility
        # - Tweet quality
        # - Time decay
        # - Influence factor
        # - Sentiment alignment
        retweet_prob = (
            self.susceptibility * 0.3 +  # Base susceptibility
            tweet_quality * 0.2 +         # Content quality
            influence_factor * 0.2 +      # Influence factor
            sentiment_alignment * 0.1 +   # Sentiment alignment
            time_decay * 0.2              # Time decay
        )
        
        # Ensure probability is in [0, 1]
        retweet_prob = max(0, min(1, retweet_prob))
        
        # Decision
        if np.random.random() < retweet_prob:
            self.retweeted = True
            self.time_retweeted = time_step
            return True
        
        return False
    
    def get_influenced_followers(self, all_agents, max_followers=100):
        """
        Get a subset of followers who will see this agent's retweet.
        
        Args:
            all_agents (list): List of all RetweetAgent objects
            max_followers (int): Maximum number of followers to return
        
        Returns:
            list: Subset of followers who will see the retweet
        """
        if self.retweeted is None:
            return []
        
        # Get followers (agents who follow this agent)
        followers = [agent for agent in all_agents if self in agent.follows]
        
        # Limit to those who haven't seen the tweet yet
        unseen_followers = [agent for agent in followers if not agent.seen]
        
        # If there are too many followers, sample based on activity level
        # (agents with higher influence are more likely to check their feeds)
        if len(unseen_followers) > max_followers:
            # Calculate selection probabilities based on influence
            influence_scores = np.array([f.influence for f in unseen_followers])
            probabilities = influence_scores / influence_scores.sum()
            
            # Sample followers based on probabilities
            return np.random.choice(
                unseen_followers, 
                size=max_followers, 
                replace=False,
                p=probabilities
            )
        
        return unseen_followers 