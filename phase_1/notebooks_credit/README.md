## First Application - Credit Risk Classification

### 1. Dataset:
- Data obtained from UCI German Credit Dataset
- Split to train (80%-90%) and test set (10%-20%)
- Features: all related to customer profile
- Target: determine whether a customer will default or not

### 2. Modeling:
- Feature engineering by category encoder and standard normalization
- KAN network learnt only from training data
- Cross-validation by 5-fold CV with stratified sampling

### 3. Benchmark:
- Naive prediction: use the most popular class to be the prediction of all 
- Conventional models: linear regression, ridge, lasso, SVM and MLP