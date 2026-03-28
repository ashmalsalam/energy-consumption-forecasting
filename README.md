# Multivariate Time-Series Energy Consumption Forecasting
 
**Regularized Linear Regression, Random Forest & Gradient Boosting with Temporal Cross-Validation** 
---
 
## 🔍 Overview
 
A complete end-to-end machine learning pipeline for short-term campus energy consumption forecasting using multivariate time-series data. The project covers synthetic data generation, comprehensive EDA, feature engineering, model development, hyperparameter tuning, and bias-variance analysis — all with temporal integrity preserved through TimeSeriesSplit cross-validation.
 
---
 
## 📊 Results
 
| Model | R² | Adj. R² | MAE | RMSE | MAPE (%) |
|---|---|---|---|---|---|
| Naive Baseline | 0.3818 | 0.3748 | 14.50 | 18.36 | 4.00 |
| **Ridge Regression** | **0.8094** | **0.8073** | **8.10** | **10.19** | **2.24** |
| Lasso Regression | 0.8067 | 0.8045 | 8.15 | 10.27 | 2.25 |
| Gradient Boosting | 0.7956 | 0.7932 | 8.42 | 10.56 | 2.32 |
| Random Forest | 0.7866 | 0.7842 | 8.62 | 10.78 | 2.38 |
 
**Ridge Regression** achieved the best test-set performance — R² of 0.8094 and RMSE of 10.19 kWh — substantially outperforming the naive baseline (R² = 0.38).
 
---
 
## 📁 Dataset
 
- **Type:** Synthetic (NumPy + Pandas, seed = 42)
- **Period:** 2025-01-01 to 2025-12-31 (hourly)
- **Records:** 8,760 raw → 8,592 after lag feature processing
- **Features:** 20 (after feature engineering)
- **Target:** `energy_kwh` — hourly campus energy consumption (range: ~295–450 kWh)
 
### Key Features
 
| Feature | Description |
|---|---|
| `temperature`, `humidity`, `wind_speed` | Meteorological variables with seasonal cycles |
| `occupancy` | Building occupancy fraction (weekday 8–18h) |
| `exam_week` | Binary flag for 3 randomly assigned exam weeks |
| `lag_1`, `lag_24`, `lag_168` | Lag features capturing short-term, daily, weekly autocorrelation |
| `rolling_mean_24/168` | Rolling mean over 24h and 168h windows |
| `hour_sin/cos`, `day_sin/cos` | Cyclical encoding of temporal features |
 
---
 
## 🔬 Methodology
 
### Exploratory Data Analysis
- Full-year and one-week time series plots
- Correlation heatmap
- ACF and PACF analysis (48 lags)
- Augmented Dickey-Fuller (ADF) stationarity test → **p-value = 8.26 × 10⁻¹⁶** (stationary)
- Variance Inflation Factor (VIF) — multicollinearity detected in meteorological features (humidity VIF = 88.79)
- Classical seasonal decomposition (24h and 168h periods) + STL decomposition
- Z-score outlier detection → only 5 outliers out of 8,592 records (0.06%)
 
### Feature Engineering
- Lag features (1h, 24h, 168h) to encode autocorrelation
- Rolling statistics (mean and std) over 24h and 168h windows
- Sine/cosine cyclical encoding for `hour` and `day_of_week`
- Chronological 80/20 train-test split (no random splitting — prevents temporal leakage)
 
### Models
- **Naive Persistence Baseline** — predicts using lag_1
- **Ridge Regression** — L2 regularization, handles multicollinearity
- **Lasso Regression** — L1 regularization, implicit feature selection
- **Random Forest Regressor** — bootstrap aggregation ensemble
- **Gradient Boosting Regressor** — sequential residual-fitting ensemble
 
### Hyperparameter Tuning
All models tuned using `GridSearchCV` + `TimeSeriesSplit(n_splits=5)` — preserves chronological order, prevents data leakage.
 
| Model | Best Parameters |
|---|---|
| Ridge | alpha = 0.1 |
| Lasso | alpha = 0.1 |
| Random Forest | n_estimators=200, max_depth=10 |
| Gradient Boosting | n_estimators=200, lr=0.05, max_depth=3 |
 
---
 
## 🧠 Key Findings
 
- **Ridge and Lasso outperformed tree ensembles** — the underlying data-generating process is predominantly linear with additive noise, making additional complexity unnecessary
- **lag_168 (weekly lag)** is the most predictive feature for ensemble models, capturing both daily and weekly seasonality
- **Occupancy and exam_week** are the dominant drivers in linear models
- **Severe multicollinearity** among meteorological features (VIF > 50) justifies regularization over OLS
- **Residuals** are approximately normally distributed across all models — model assumptions satisfied
- **Bias-variance analysis** confirms Ridge and Lasso provide the best trade-off between accuracy and stability
 
---
 
## 🛠️ Tech Stack
 
`Python` `NumPy` `Pandas` `Scikit-learn` `Statsmodels` `Matplotlib` `Seaborn` `SciPy`
 
---
 
## 📂 Repository Structure
 
```
energy-consumption-forecasting/
│
├── Energy_Consumption_Forecasting.ipynb       # Full notebook — EDA, modelling, evaluation
├── Project_Report.pdf      # Detailed project report
└── README.md
```
 
---
 
## 🚀 How to Run
 
```bash
# Clone the repository
git clone https://github.com/ashmalsalam/energy-consumption-forecasting.git
cd energy-consumption-forecasting
 
# Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn statsmodels scipy
 
# Run the notebook
jupyter notebook Energy_Consumption_Forecasting.ipynb
```
 
---
 
## 📈 Future Work
 
- Apply framework to real-world campus energy meter data
- Incorporate LSTM and Transformer architectures for long-range temporal dependencies
- Explore multi-step ahead forecasting (24h, 48h) for operational scheduling
- Integrate real weather APIs and academic calendar data
- Develop an online learning pipeline for continuous retraining
 
---
 
## 👤 Author
 
**Ashmal Abdussalam P T**
