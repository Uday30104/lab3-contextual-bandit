# User Classification for Contextual Bandits

This project implements a user classification pipeline that acts as a **context detector** for a contextual bandit system.  
Users are classified into three categories (`user_1`, `user_2`, `user_3`) based on behavioral, transactional, and device-level features.

---

## Problem Overview
Given historical user interaction data, the goal is to:
- Preprocess and clean the dataset
- Train a classifier to predict user categories
- Evaluate model performance on a validation set
- Predict user context for unseen test users

---

## Dataset
Each user record contains:
- Behavioral features (clicks, session duration, engagement)
- Transactional features (purchase amount, cart value)
- Device/system features (battery, browser version, network jitter)

Target variable:
- `label` ∈ {`user_1`, `user_2`, `user_3`}

---

## Methodology
- Dropped identifier columns (`user_id`)
- Handled missing values using median (numerical) and mode (categorical)
- Encoded categorical features with `OrdinalEncoder`
- Used an 80/20 train–validation split

---

## Models
- **Logistic Regression** (baseline, with feature scaling)  
  - Validation accuracy: ~82%
- **Decision Tree Classifier** (final model)  
  - Captures non-linear user behavior  
  - Validation accuracy: **~86%**

The Decision Tree was selected as the final context detector due to better overall performance.

---

## Evaluation
Models were evaluated using `classification_report` (precision, recall, F1-score).  
The Decision Tree showed improved recall across all user categories.

---

## Context Detection
The trained classifier is applied to unseen test users to infer user categories, which serve as context inputs for downstream bandit algorithms.

---