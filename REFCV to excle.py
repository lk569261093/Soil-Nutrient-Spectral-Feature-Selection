import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# 1. 读取 Excel 数据
file_path = "SOM.xlsx"
df = pd.read_excel(file_path, engine='openpyxl')

# 2. 数据预处理
X = df.iloc[:, 2:] # 提取光谱数据
y = df.iloc[:, 1] # 提取土壤养分含量数据

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 定义回归模型列表
models = [
("Linear Regression", LinearRegression()),
("Ridge Regression", Ridge(alpha=1.0)),
("Lasso Regression", Lasso(alpha=1.0)),
("Support Vector Regression", SVR(kernel='linear', C=1.0)),
("Random Forest Regressor", RandomForestRegressor(n_estimators=100)),
("Gradient Boosting Regressor", GradientBoostingRegressor(n_estimators=100, learning_rate=0.1)),
("XGBoost Regressor", XGBRegressor(n_estimators=100, learning_rate=0.1)),
("LightGBM Regressor", LGBMRegressor(n_estimators=100, learning_rate=0.1))
]

# 创建一个空的DataFrame用于保存特征选择结果
selected_features_df = pd.DataFrame(columns=['Model', 'Selected Features'])

# 对每个模型执行特征选择并将结果保存到DataFrame中
for model_name, model in models:
    selector = SelectFromModel(model, max_features=10)
    selector.fit(X_scaled, y)

    selected_features = X.columns[selector.get_support()]
    #selected_features_str = ', '.join(selected_features.tolist())
    selected_features_str = ', '.join(map(str, selected_features.tolist()))
    #selected_features_df = selected_features_df.append({'Model': model_name, 'Selected Features': selected_features_str}, ignore_index=True)
    new_row = {'Model': model_name, 'Selected Features': selected_features_str}
    selected_features_df = pd.concat([selected_features_df, pd.DataFrame(new_row, index=[0])], ignore_index=True)


# 输出结果到Excel文件中
output_file_path = file_path.replace(".xlsx", "_特征选择.xlsx")
selected_features_df.to_excel(output_file_path, index=False, engine='openpyxl')
