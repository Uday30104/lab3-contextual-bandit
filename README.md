# Lab 3: Contextual Multi-Armed Bandit for News Recommendation

This project implements a **Contextual Multi-Armed Bandit (CMAB)** based news recommendation system.  
The system combines **user classification** with **reinforcement learning bandit strategies** to recommend news articles that maximize user engagement.

---

## Problem Overview

The environment is modeled as a **Contextual Bandit** problem:

- **Contexts:** 3 user types (`user_1`, `user_2`, `user_3`)
- **Arms:** 4 news categories per context  
  (`ENTERTAINMENT`, `EDUCATION`, `TECH`, `CRIME`)
- **Total Arms:** 12 (3 contexts × 4 categories)

The goal is to **maximize expected user engagement (reward)** by selecting the optimal news category for each user.

Rewards are obtained using the **provided `rlcmab_sampler` utility**, which simulates unknown reward distributions and is used strictly as provided.

---

## Datasets

### User Data
- `train_users.csv`: Used to train a user classification model
- `test_users.csv`: Used for context prediction during recommendation

Each user is classified into one of three contexts (`user_1`, `user_2`, `user_3`).

### News Articles
- `news_articles.csv`: Contains news articles with metadata such as headline, category, and link

Only the following categories are used as bandit arms:
- `ENTERTAINMENT`
- `EDUCATION`
- `TECH`
- `CRIME`

---

## Methodology

### 1. Data Preprocessing
- Removed identifier columns
- Handled missing values (median for numerical, mode for categorical)
- Encoded categorical features
- Applied feature scaling
- Ensured identical preprocessing for training and test data

---

### 2. User Classification (Context Detection)
- Trained a supervised classification model
- 80/20 train–validation split
- Evaluated using `classification_report`
- The trained classifier predicts the **user context**, which is used as input to the bandit system

---

### 3. Contextual Bandit Algorithms

A **separate bandit model is trained for each user context** using the following strategies:

#### Epsilon-Greedy
- Explores with probability ε, exploits otherwise
- Tested with multiple ε values (0.05, 0.1, 0.2)
- Lower ε achieved higher rewards in this stationary environment

#### Upper Confidence Bound (UCB)
- Uses optimism-based exploration
- Tested with different values of exploration parameter C
- Demonstrated robust and consistently strong performance

#### Softmax
- Uses probabilistic action selection
- Fixed temperature parameter τ = 1
- Achieved competitive performance by gradually reducing exploration

All algorithms were trained for a horizon of **T = 10,000 steps** using the provided sampler.

---

## Recommendation Engine (End-to-End System)

The final recommendation pipeline works as follows:

1. **Classify User:** Predict user context using the trained classifier
2. **Select Category:** Choose the optimal news category using the trained bandit policy (UCB)
3. **Recommend Article:** Randomly sample an article from the selected category
4. **Output:** Predicted user context, recommended category, headline, and link

---

## Evaluation & Results

### Classification Performance
- Evaluated using `classification_report` on validation data
- Achieved strong and balanced performance across all user classes

### Reinforcement Learning Performance
- **UCB** achieved the highest average reward across all contexts
- **Softmax** performed better than ε-greedy but slightly below UCB
- **Epsilon-Greedy** showed sensitivity to the exploration rate

### Key Observations
- Lower exploration is beneficial in stationary environments
- UCB provides the best balance between exploration and exploitation
- Different user contexts exhibit different reward characteristics

## Conclusion

This project demonstrates how **Contextual Multi-Armed Bandits** can effectively personalize recommendations by combining supervised learning and reinforcement learning.  
Among the evaluated strategies, **UCB** consistently achieved the best performance and was selected for deployment in the recommendation engine.