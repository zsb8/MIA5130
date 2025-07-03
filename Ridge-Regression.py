# Ridge Regression Demand Forecasting with Store-Product Composite ID

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# === 1. Load dataset ===
df = pd.read_csv("sales_data.csv")

# === 2. Convert Date ===
df['Date'] = pd.to_datetime(df['Date'])

# === 3. Create composite ID ===
# === Each time series is uniquely identified by Store ID + Product ID.
# === It is necessary because the dataset has Store ID and Product ID as follows:
# === Stored ID | Product ID
# ===   S001    |    P0001
# ===   S001    |    P0002
# ===   S002    |    P0001
# ===   S002    |    P0002
df['Store_Product_ID'] = df['Store ID'].astype(str) + '_' + df['Product ID'].astype(str)

# === 4. Sort for lag features ===
# === Ensures lag and rolling computations are applied in time order within each product-store.
df = df.sort_values(by=['Store_Product_ID', 'Date'])

# === 5. Lag and rolling features ===
# === Creates historical features (lags) and 7-day/14-day rolling averages to capture trends and seasonality.
df['Lag_1_Demand'] = df.groupby('Store_Product_ID')['Demand'].shift(1)
df['Lag_2_Demand'] = df.groupby('Store_Product_ID')['Demand'].shift(2)
df['Lag_3_Demand'] = df.groupby('Store_Product_ID')['Demand'].shift(3)
df['Rolling_7_Demand'] = df.groupby('Store_Product_ID')['Demand'].shift(1).rolling(window=7).mean()
df['Rolling_14_Demand'] = df.groupby('Store_Product_ID')['Demand'].shift(1).rolling(window=14).mean()

# === 6. Backorder and stockout risk ===
# === Binary features that help the model learn from supply-demand imbalances.
df['Backorder'] = (df['Units Ordered'] > df['Units Sold']).astype(int)
df['Stockout_Risk'] = (df['Inventory Level'] < df['Units Ordered']).astype(int)

# === 7. Date features + seasonality ===
# === Extracts time-based signals, including cyclical patterns like seasonal sinusoids for DayOfYear.
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['Weekday'] = df['Date'].dt.weekday
df['Quarter'] = df['Date'].dt.quarter
df['DayOfYear'] = df['Date'].dt.dayofyear
df['Is_Weekend'] = df['Weekday'].isin([5, 6]).astype(int)
df['Sin_DayOfYear'] = np.sin(2 * np.pi * df['DayOfYear'] / 365)
df['Cos_DayOfYear'] = np.cos(2 * np.pi * df['DayOfYear'] / 365)

# === 8. Encode categoricals ===
# === Transforms regions, weather, category, and seasonality into numeric columns the model can use.
df = pd.get_dummies(df, columns=['Region', 'Weather Condition', 'Seasonality', 'Category'], drop_first=True)

# === 9. Feature interactions ===
# === Combines features that interact logically
df['Price_Discount'] = df['Price'] * df['Discount']
df['Price_vs_Competitor'] = df['Price'] - df['Competitor Pricing']
df['Promo_Discount'] = df['Promotion'] * df['Discount']

# === 10. Drop rows with NaNs ===
# === Removes rows with missing values after lagging and rolling.
print(f"The number of rows before dropping rows with NaNs : {len(df)}")
df.dropna(inplace=True)
print(f"The number of rows after dropping rows with NaNs : {len(df)}")

# === 11. Normalize demand by composite ID ===
# === Standardizes Demand within each product-store group to help the model converge better.
df['Demand_scaled'] = df.groupby('Store_Product_ID')['Demand'].transform(
    lambda x: (x - x.mean()) / x.std()
)

# === 12. Set target and features ===
# === Sets target (y) and predictors (X) for training.
y = df['Demand_scaled']

feature_cols = [
    'Price', 'Discount', 'Competitor Pricing', 'Promotion', 'Inventory Level',
    'Units Ordered', 'Month', 'Day', 'Weekday', 'Quarter', 'DayOfYear', 'Is_Weekend',
    'Sin_DayOfYear', 'Cos_DayOfYear', 'Backorder', 'Stockout_Risk', 'Epidemic',
    'Lag_1_Demand', 'Lag_2_Demand', 'Lag_3_Demand', 'Rolling_7_Demand', 'Rolling_14_Demand',
    'Price_Discount', 'Price_vs_Competitor', 'Promo_Discount'
]

feature_cols += [col for col in df.columns if
                 col.startswith('Region_') or
                 col.startswith('Weather Condition_') or
                 col.startswith('Seasonality_') or
                 col.startswith('Category_')]

X = df[feature_cols]

# === Combines polynomial feature expansion, standard scaling, and RidgeCV in a clean workflow.
# === PolynomialFeatures(degree=2) lets the model learn interactions/quadratic effects.
# Sometimes I have to decrease degree to 1 because my computer doesn't have enough memory to run the code
# === 13. TimeSeriesSplit ===
tscv = TimeSeriesSplit(n_splits=5)

# === 14. RidgeCV Pipeline ===
pipeline = Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('scaler', StandardScaler()),
    ('ridge', RidgeCV(alphas=[0.001, 0.01, 0.1, 1.0, 10.0], scoring='r2', cv=tscv))
])

# === Trains the full pipeline with cross-validated Ridge regression.
pipeline.fit(X, y)

# === Gets predictions, then reverts them from scaled units back to actual demand values.
# === 15. Prediction ===
y_pred_scaled = pipeline.predict(X)

# === 16. Inverse scale back to original demand ===
preds_df = df[['Store_Product_ID', 'Demand', 'Date']].copy()
preds_df['y_pred_scaled'] = y_pred_scaled

means = df.groupby('Store_Product_ID')['Demand'].transform('mean')
stdevs = df.groupby('Store_Product_ID')['Demand'].transform('std')

preds_df['y_pred'] = preds_df['y_pred_scaled'] * stdevs + means

# === Add 95% prediction interval ===
# === Uses standard deviation of residuals to create a confidence band (±1.96σ covers ~95%).
residual_std = (preds_df['Demand'] - preds_df['y_pred']).std()
preds_df['y_upper'] = preds_df['y_pred'] + 1.96 * residual_std
preds_df['y_lower'] = preds_df['y_pred'] - 1.96 * residual_std

# === 17. Evaluation ===
r2 = r2_score(preds_df['Demand'], preds_df['y_pred'])
mae = mean_absolute_error(preds_df['Demand'], preds_df['y_pred'])
rmse = np.sqrt(mean_squared_error(preds_df['Demand'], preds_df['y_pred']))
actual_std = preds_df['Demand'].std()

print(f"R² Score ( % variance explained ): {r2:.4f}")
print(f"MAE ( average absolute error ): {mae:.4f}")
print(f"RMSE ( error magnitude (in same units as demand) ): {rmse:.4f}")
print(f"Std Dev of Actual Demand: {actual_std:.4f}")
print(f"Std Dev of Residuals: {residual_std:.4f}")


# === 18. Residual Plot ===
# === Visual check for bias: residuals should be centered around 0 and evenly spread.
plt.figure(figsize=(8,6))
plt.scatter(preds_df['y_pred'], preds_df['Demand'] - preds_df['y_pred'], alpha=0.3)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted Demand")
plt.ylabel("Residuals (Actual - Predicted)")
plt.title("Residual Plot")
plt.show()

# === 19. Time Series Plot with Prediction Interval ===
# === Actual vs Predicted Demand over time
# === Shaded area = 95% prediction interval
preds_df = preds_df.sort_values('Date')

plt.figure(figsize=(12,6))
plt.plot(preds_df['Date'], preds_df['Demand'], label='Actual Demand', alpha=0.7)
plt.plot(preds_df['Date'], preds_df['y_pred'], label='Predicted Demand', linestyle='--', alpha=0.7)
plt.fill_between(preds_df['Date'], preds_df['y_lower'], preds_df['y_upper'], color='gray', alpha=0.2, label='95% Interval')
plt.title("Actual vs Predicted Demand with 95% Prediction Interval")
plt.xlabel("Date")
plt.ylabel("Demand")
plt.legend()
plt.grid(True)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()