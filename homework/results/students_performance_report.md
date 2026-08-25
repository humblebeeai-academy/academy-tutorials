# Math Score Prediction from Gender - Linear Regression

## Dataset
- Source: Kaggle - Students Performance in Exams
- 1000 rows, 8 columns (only `gender` and `math score` are used in this task)
- Cleaning: no missing values or duplicates found; the `gender` text column was encoded to 0/1 (female=0, male=1)

## Model
- Type: Simple Linear Regression (1 feature: gender_encoded)
- Train/Test split: 80/20 (random_state=42)

## Results

| Metric | Value |
|---|---|
| Slope (m) | 4.59 |
| Intercept (b) | 64.32 |
| MAE | 12.05 |
| MSE | 236.59 |
| RMSE | 15.38 |
| R² | 0.0278 |

## Conclusion
Each unit increase in gender_encoded is associated with roughly a 4.59-point change in math score. On the test set, the average error (RMSE) is 15.38, and R² = 0.0278 means the model explains 2.8% of the variance in math score.
