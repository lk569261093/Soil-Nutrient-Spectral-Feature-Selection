import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# 读取 Excel 数据
data = pd.read_excel('SOM.xlsx')
X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values

# 数据预处理
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 使用随机森林回归器计算特征重要性
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_scaled, y)
importances = rf_regressor.feature_importances_

# 对特征重要性进行排序，并获取前15个最重要特征的索引
top_15_indices = np.argsort(importances)[-15:][::-1]

print("Top 15 most important spectral features:")
print(top_15_indices)
