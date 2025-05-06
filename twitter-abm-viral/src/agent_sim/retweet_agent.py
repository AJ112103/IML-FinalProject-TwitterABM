import numpy as np

class RetweetAgent:
    
    def __init__(self, followers=100, influence=None, susceptibility=None, sentiment_bias=None):

        # Basic properties
        self.followers = max(1, int(followers))

        if influence is not None:
            self.influence = np.clip(influence, 0, 1)
        else:
            base_influence = 0.1 + 0.4 * np.log1p(self.followers) / np.log1p(10000)
            self.influence = np.clip(base_influence, 0.1, 0.9)

        if susceptibility is not None:
            self.susceptibility = np.clip(susceptibility, 0, 1)
        else:
            self.susceptibility = np.clip(np.random.beta(2, 3), 0, 1)

        if sentiment_bias is not None:
            self.sentiment_bias = np.clip(sentiment_bias, -1, 1)
        else:
            self.sentiment_bias = np.clip(np.random.normal(0, 0.3), -1, 1)
        
        # State variables
        self.seen = False             
        self.retweeted = None         
        self.time_seen = None        
        self.time_retweeted = None
    
    def reset(self):

        self.seen = False
        self.retweeted = None
        self.time_seen = None
        self.time_retweeted = None
    
    def see_tweet(self, time_step):

        if not self.seen:
            self.seen = True
            self.time_seen = time_step
            return True
        return False
    
    def decide_retweet(self, tweet_quality, tweet_sentiment, time_step):

        if not self.seen or self.retweeted is not None:
            return False

        time_elapsed = time_step - self.time_seen
        time_decay = np.exp(-0.1 * time_elapsed)  
        
        influence_factor = self.influence

        normalized_sentiment = tweet_sentiment / 5

        sentiment_alignment = 1 - abs(normalized_sentiment - self.sentiment_bias)

        retweet_prob = (
            self.susceptibility * 0.3 +
            tweet_quality * 0.2 +        
            influence_factor * 0.2 +      
            sentiment_alignment * 0.1 +   
            time_decay * 0.2              
        )

        retweet_prob = max(0, min(1, retweet_prob))

        if np.random.random() < retweet_prob:
            self.retweeted = True
            self.time_retweeted = time_step
            return True
        
        return False
    
    def get_influenced_followers(self, all_agents, max_followers=100):

        if self.retweeted is None:
            return []

        followers = [agent for agent in all_agents if self in agent.follows]

        unseen_followers = [agent for agent in followers if not agent.seen]

        if len(unseen_followers) > max_followers:
            influence_scores = np.array([f.influence for f in unseen_followers])
            probabilities = influence_scores / influence_scores.sum()

            return np.random.choice(
                unseen_followers, 
                size=max_followers, 
                replace=False,
                p=probabilities
            )
        
        return unseen_followers 