import numpy as np

# Simulated click-through rate (CTR) for each ad
true_ctr = np.array([0.1, 0.15, 0.2, 0.25, 0.3])

class EpsilonGreedyBandit:
    def __init__(self, num_arms, epsilon):
        self.num_arms = num_arms
        self.epsilon = epsilon
        self.action_counts = np.zeros(num_arms)
        self.q_estimates = np.zeros(num_arms)
    
    def choose_arm(self):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.num_arms)  # Explore
        else:
            return np.argmax(self.q_estimates)  # Exploit
    
    def update(self, chosen_arm, reward):
        self.action_counts[chosen_arm] += 1
        alpha = 1 / self.action_counts[chosen_arm]
        self.q_estimates[chosen_arm] += alpha * (reward - self.q_estimates[chosen_arm])

def simulate_bidding():
    num_rounds = 1000
    total_budget = 1000
    bid_prices = np.linspace(0.1, 1.0, 5)  # Simulated bid prices
    
    bandit = EpsilonGreedyBandit(num_arms=len(bid_prices), epsilon=0.1)
    total_clicks = 0
    total_spent = 0
    
    for _ in range(num_rounds):
        chosen_arm = bandit.choose_arm()
        bid_price = bid_prices[chosen_arm]
        reward = np.random.binomial(1, true_ctr[chosen_arm])
        
        if bid_price <= total_budget:
            total_budget -= bid_price
            total_spent += bid_price
            total_clicks += reward
        
        bandit.update(chosen_arm, reward)
    
    ctr = total_clicks / num_rounds
    cpc = total_spent / total_clicks if total_clicks > 0 else 0
    
    return ctr, cpc

ctr, cpc = simulate_bidding()

print("Click-through Rate (CTR): {:.2%}".format(ctr))
print("Cost per Click (CPC): ${:.2f}".format(cpc))

