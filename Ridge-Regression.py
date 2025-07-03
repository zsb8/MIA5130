# Ridge Regression Demand Forecasting with Store-Product Composite ID

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

df = pd.read_csv("sales_data.csv")
df['Date'] = pd.to_datetime(df['Date'])
# Create composite ID
# === Stored ID | Product ID
# ===   S001    |    P0001
# ===   S001    |    P0002
# ===   S002    |    P0001
# ===   S002    |    P0002
df['Store_Product_ID'] = df['Store ID'].astype(str) + '_' + df['Product ID'].astype(str)
# ort for lag features ===
df = df.sort_values(by=['Store_Product_ID', 'Date'])
# Creates historical features (lags) and 7-day/14-day rolling averages to capture trends and seasonality
df['Lag_1_Demand'] = df.groupby('Store_Product_ID')['Demand'].shift(1)
df['Lag_2_Demand'] = df.groupby('Store_Product_ID')['Demand'].shift(2)
df['Lag_3_Demand'] = df.groupby('Store_Product_ID')['Demand'].shift(3)
df['Rolling_7_Demand'] = df.groupby('Store_Product_ID')['Demand'].shift(1).rolling(window=7).mean()
df['Rolling_14_Demand'] = df.groupby('Store_Product_ID')['Demand'].shift(1).rolling(window=14).mean()
# Binary features that help the model learn from supply-demand imbalances
df['Backorder'] = (df['Units Ordered'] > df['Units Sold']).astype(int)
df['Stockout_Risk'] = (df['Inventory Level'] < df['Units Ordered']).astype(int)
# Extracts time-based signals
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['Weekday'] = df['Date'].dt.weekday
df['Quarter'] = df['Date'].dt.quarter
df['DayOfYear'] = df['Date'].dt.dayofyear
df['Is_Weekend'] = df['Weekday'].isin([5, 6]).astype(int)
df['Sin_DayOfYear'] = np.sin(2 * np.pi * df['DayOfYear'] / 365)
df['Cos_DayOfYear'] = np.cos(2 * np.pi * df['DayOfYear'] / 365)

# One-hot, add columnbs such as 'Region_North', 'Region_South', 'Region_West', only True or Falseq value.
# Nots: We use drop_first=True to avoid Dummy Variable Trap
df = pd.get_dummies(df, columns=['Region', 'Weather Condition', 'Seasonality', 'Category'], drop_first=True)
# Combines features that interact logically
df['Price_Discount'] = df['Price'] * df['Discount']
df['Price_vs_Competitor'] = df['Price'] - df['Competitor Pricing']
df.dropna(inplace=True)
# Standardizes Demand within each product-store group to help the model converge better.
df['Demand_scaled'] = df.groupby('Store_Product_ID')['Demand'].transform(
    lambda x: (x - x.mean()) / x.std()
)
# Sets target (y) and predictors (X) for training.
y = df['Demand_scaled']

feature_cols = [
    'Price', 'Discount', 'Competitor Pricing', 'Promotion', 'Inventory Level',
    'Units Ordered', 'Month', 'Day', 'Weekday', 'Quarter', 'DayOfYear', 'Is_Weekend',
    'Sin_DayOfYear', 'Cos_DayOfYear', 'Backorder', 'Stockout_Risk', 'Epidemic',
    'Lag_1_Demand', 'Lag_2_Demand', 'Lag_3_Demand', 'Rolling_7_Demand', 'Rolling_14_Demand',
    'Price_Discount', 'Price_vs_Competitor'
]

feature_cols += [col for col in df.columns if
                 col.startswith('Region_') or
                 col.startswith('Weather Condition_') or
                 col.startswith('Seasonality_') or
                 col.startswith('Category_')]
X = df[feature_cols]

# Split Train and Test Dataset
X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
    X, y, df, test_size=0.2, shuffle=False
)

""" 
RidgeCV Pipeline
Combines polynomial feature expansion, standard scaling, and RidgeCV in a clean workflow.
PolynomialFeatures lets the model learn interactions/quadratic effects. 
degree=2: R2=0.6760, MAE=20.9665, RESE: 27.5440
degree=3: R2=0.7397, MAE=18.0209, RESE: 23.9830
"""
pipeline = Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('scaler', StandardScaler()),
    ('ridge', RidgeCV(alphas=[0.001, 0.01, 0.1, 1.0, 10.0], scoring='r2', cv=3))
])
# Trains the full pipeline with cross-validated Ridge regression.
pipeline.fit(X_train, y_train)
# Gets predictions, then reverts them from scaled units back to actual demand values.
y_pred_train_scaled = pipeline.predict(X_train)
y_pred_test_scaled = pipeline.predict(X_test)

# Restore to original demand
means_train = df_train.groupby('Store_Product_ID')['Demand'].transform('mean')
stdevs_train = df_train.groupby('Store_Product_ID')['Demand'].transform('std')
means_test = df_test.groupby('Store_Product_ID')['Demand'].transform('mean')
stdevs_test = df_test.groupby('Store_Product_ID')['Demand'].transform('std')

preds_train_df = df_train[['Store_Product_ID', 'Demand', 'Date']].copy()
preds_train_df['y_pred_scaled'] = y_pred_train_scaled
preds_train_df['y_pred'] = preds_train_df['y_pred_scaled'] * stdevs_train + means_train

preds_test_df = df_test[['Store_Product_ID', 'Demand', 'Date']].copy()
preds_test_df['y_pred_scaled'] = y_pred_test_scaled
preds_test_df['y_pred'] = preds_test_df['y_pred_scaled'] * stdevs_test + means_test

# Evaluation
r2_train = r2_score(preds_train_df['Demand'], preds_train_df['y_pred'])
mae_train = mean_absolute_error(preds_train_df['Demand'], preds_train_df['y_pred'])
rmse_train = np.sqrt(mean_squared_error(preds_train_df['Demand'], preds_train_df['y_pred']))

r2_test = r2_score(preds_test_df['Demand'], preds_test_df['y_pred'])
mae_test = mean_absolute_error(preds_test_df['Demand'], preds_test_df['y_pred'])
rmse_test = np.sqrt(mean_squared_error(preds_test_df['Demand'], preds_test_df['y_pred']))


print(f"[Train Dataset] R²: {r2_train:.4f}, MAE: {mae_train:.4f}, RMSE: {rmse_train:.4f}")
print(f"[Test Dataset] R²: {r2_test:.4f}, MAE: {mae_test:.4f}, RMSE: {rmse_test:.4f}")
actual_std_test = preds_test_df['Demand'].std()
print(f"Std Dev of Actual Demand (Test Set): {actual_std_test:.4f}")
residual_std_test = (preds_test_df['Demand'] - preds_test_df['y_pred']).std()
print(f"Std Dev of Residuals (Test Set): {residual_std_test:.4f}")

# Residual Plot , test dataset
plt.figure(figsize=(8,6))
plt.scatter(preds_test_df['y_pred'], preds_test_df['Demand'] - preds_test_df['y_pred'], alpha=0.3)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted Demand")
plt.ylabel("Residuals (Actual - Predicted)")
plt.title("Residual Plot")
plt.show()

# Time Series Plot with Prediction Interval. Actual vs Predicted Demand over time. Shaded area = 95% prediction interval
preds_test_df = preds_test_df.sort_values('Date')
plt.figure(figsize=(12,6))
plt.plot(preds_test_df['Date'], preds_test_df['Demand'], label='Actual Demand', alpha=0.7)
plt.plot(preds_test_df['Date'], preds_test_df['y_pred'], label='Predicted Demand', linestyle='--', alpha=0.7)
plt.fill_between(preds_test_df['Date'], preds_test_df['y_pred'] - 1.96 * (preds_test_df['Demand'] - preds_test_df['y_pred']).std(),
                 preds_test_df['y_pred'] + 1.96 * (preds_test_df['Demand'] - preds_test_df['y_pred']).std(),
                 color='gray', alpha=0.2, label='95% Interval')
plt.title("Actual vs Predicted Demand with 95% Prediction Interval (Test Set)")
plt.xlabel("Date")
plt.ylabel("Demand")
plt.legend()
plt.grid(True)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
