import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# Read Excel data
data = pd.read_excel('SOM.xlsx')
X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values

# Data preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Calculate feature importance using a random forest regressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_scaled, y)
importances = rf_regressor.feature_importances_

# Sort the feature importance and obtain the indices of the top 15 most important features.
top_15_indices = np.argsort(importances)[-15:][::-1]

print("Top 15 most important spectral features:")
print(top_15_indices)
