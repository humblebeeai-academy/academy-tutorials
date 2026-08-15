# Salary Prediction - Linear Regression

## Dataset
- Manba: Kaggle - Salary Dataset (Simple Linear Regression)
- 30 qator, 2 ustun: YearsExperience, Salary
- Tozalash: keraksiz indeks ustuni (`Unnamed: 0`) olib tashlandi; missing value va dublikat topilmadi

## Model
- Turi: Simple Linear Regression (1 xususiyat: YearsExperience)
- Train/Test split: 80/20 (random_state=42)

## Natijalar

| Metrika | Qiymat |
|---|---|
| Slope (m) | 9423.82 |
| Intercept (b) | 24380.20 |
| MAE | 6286.45 |
| MSE | 49830096.86 |
| RMSE | 7059.04 |
| R² | 0.9024 |

## Xulosa
Model har bir qo’shimcha ish tajribasi yiliga taxminan 9424 birlik maosh o’sishini bashorat qiladi. Test to’plamida o’rtacha xato (RMSE) 7059 bo’lib, R² = 0.9024 ekani modelning maosh o’zgarishining 90.2% ini tushuntira olishini ko’rsatadi - bu juda yaxshi mos kelish darajasi.
