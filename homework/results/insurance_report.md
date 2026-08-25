# Insurance Charges Prediction from BMI - Linear Regression

## Dataset
- Source: Kaggle - Medical Cost Personal Datasets (insurance.csv)
- 1338 rows, 7 columns (only `bmi` and `charges` are used in this task)
- Cleaning: no missing values; 1 fully duplicated row was found and dropped

## Model
- Type: Simple Linear Regression (1 feature: bmi)
- Train/Test split: 80/20 (random_state=42)

## Results

| Metric | Value |
|---|---|
| Slope (m) | 345.17 |
| Intercept (b) | 2488.57 |
| MAE | 9891.12 |
| MSE | 174251720.52 |
| RMSE | 13200.44 |
| R² | 0.0517 |

## Outliers
- IQR upper bound on charges: 34524.78
- Outliers found: 139 out of 1337 rows (10.4%)
- Slope with outliers: 345.17 | Slope without outliers: -80.01

## Conclusion
Each unit increase in BMI is associated with roughly a 345.17-unit increase in predicted charges. On the test set, RMSE = 13200.44 and R² = 0.0517, meaning BMI alone explains only 5.2% of the variance in charges. The IQR-based outlier check found 139 high-charge outliers, which noticeably steepen the fitted line - a known weakness of least squares, since squared error gives large residuals outsized influence.
