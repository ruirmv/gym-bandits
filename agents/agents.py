import numpy as np


class Agent:
    
    def __init__(self, n_actions):
        self.n_actions = n_actions
    
    def act(self):
        pass
    
    def update(self, action, reward):
        pass


class EpsilonGreedyAgent(Agent):
    
    def __init__(self, n_actions, estimates, counts, epsilon=0.01):
        self.estimates = None
        self.counts = None
        
        if 0 > epsilon or epsilon > 1:
            raise ValueError("Epsilon must be a value between 0 and 1.")
        
        if len(estimates) != len(counts) != n_actions:
            raise ValueError("An estimate and initial count should be provided for each action. Expected np.array.")
        
        super().__init__(n_actions)
        
        self.epsilon = epsilon
        self.initial_estimates = estimates.copy()
        self.initial_counts = counts.copy()
        
        self.reset()
    
    def act(self):
        if np.random.rand() >= self.epsilon:
            distribution = np.where(self.estimates == max(self.estimates), 1, 0)
            return np.random.choice(self.n_actions, p=distribution / sum(distribution))
        else:
            return np.random.choice(self.n_actions)
    
    def update(self, action, reward):
        self.counts[action] += 1
        self.estimates[action] += 1. / self.counts[action] * (reward - self.estimates[action])
    
    def reset(self):
        self.estimates = self.initial_estimates.copy()
        self.counts = self.initial_counts.copy()


class GreedyAgent(EpsilonGreedyAgent):
    
    def __init__(self, n_actions, estimates, counts):
        super().__init__(n_actions, estimates, counts, epsilon=0)


class UCBAgent(Agent):
    
    def __init__(self, n_actions, estimates, counts, step=0, c=1):
        self.estimates = None
        self.counts = None
        self.step = None
        
        if c < 0:
            raise ValueError("The value of c must be larger than 0.")
        
        super().__init__(n_actions)
        self.initial_estimates = estimates.copy()
        self.initial_counts = counts.copy()
        self.initial_step = step
        self.c = c
        
        self.reset()
    
    def act(self):
        
        if any(self.counts == 0):
            return np.random.choice(np.nonzero(self.counts == 0)[0])
        
        bound = self.estimates + self.c * np.sqrt(np.log(self.step) / self.counts)
        
        return np.random.choice(np.nonzero(bound == max(bound))[0])
    
    def update(self, action, reward):
        self.counts[action] += 1
        self.estimates[action] += 1. / self.counts[action] * (reward - self.estimates[action])
        self.step += 1
    
    def reset(self):
        self.estimates = self.initial_estimates.copy()
        self.counts = self.initial_counts.copy()
        self.step = self.initial_step


class BanditGradientAgent(Agent):
    
    def __init__(self, n_actions, preferences, average_reward=0, n=0, alpha=0.5):
        self.n = None
        self.preferences = None
        self.average_reward = None
        self.distribution = None
        
        super().__init__(n_actions)
        
        if alpha < 0:
            raise ValueError("Alpha must be a value larger than 0.")
        elif n < 0:
            raise ValueError("N must be a value larger than 0.")
        
        self.alpha = alpha
        self.initial_n = n
        self.initial_preferences = preferences.copy()
        self.initial_average_reward = average_reward
        
        self.reset()
    
    def act(self):
        return np.random.choice(self.n_actions, p=self.distribution)
    
    def update(self, action, reward):
        reward_error = reward - self.average_reward
        discriminant = np.where(np.arange(self.n_actions) == action, 1, 0)
        
        self.preferences += self.alpha * reward_error * (discriminant - self.distribution)
        self.n += 1
        self.average_reward += 1. / self.n * reward_error
    
    def reset(self):
        self.n = self.initial_n
        self.preferences = self.initial_preferences
        self.average_reward = self.initial_average_reward
        
        distribution = np.exp(self.preferences)
        self.distribution = distribution / sum(distribution)
