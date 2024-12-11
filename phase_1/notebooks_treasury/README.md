## Second Application - Yield Curve Prediction

### 1. Dataset:
- Daily treasury par yield curve rates obtained from https://home.treasury.gov/
- Split to train (80%-90%) and test set (10%-20%)
- Feature: all yields at different maturity of the past 3-5 business days

### 2. Modeling:
- KAN network learnt only from training data
- Recursive prediction: last prediction to be feature of the next prediction
- Horizon: 5-10-20 business days
- Cross-validation by shifting 10 days forward or backward

### 3. Benchmark:
- Naive prediction: use the last observation of training set to be prediction of all future dates
- Random walk: estimate drift and volatility by historical data, and simulate forward by GBM
- Conventional models: linear regression, ridge, lasso and MLP with recursive approach
- Past research papers: Diebold & Li, regime-switching 