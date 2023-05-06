import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# Read Excel data
file_path = "SOM.xlsx"
df = pd.read_excel(file_path, engine='openpyxl')

# Data preprocessing
X = df.iloc[:, 2:] # Extract spectral data
y = df.iloc[:, 1] # Extract soil nutrient content data

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define a list of regression models
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

# Create an empty DataFrame to store the results of feature selection.
selected_features_df = pd.DataFrame(columns=['Model', 'Selected Features'])

# Perform feature selection on each model and save the results to a DataFrame.
for model_name, model in models:
    selector = SelectFromModel(model, max_features=10)
    selector.fit(X_scaled, y)

    selected_features = X.columns[selector.get_support()]
    #selected_features_str = ', '.join(selected_features.tolist())
    selected_features_str = ', '.join(map(str, selected_features.tolist()))
    #selected_features_df = selected_features_df.append({'Model': model_name, 'Selected Features': selected_features_str}, ignore_index=True)
    new_row = {'Model': model_name, 'Selected Features': selected_features_str}
    selected_features_df = pd.concat([selected_features_df, pd.DataFrame(new_row, index=[0])], ignore_index=True)


# Export the results to an Excel file.
output_file_path = file_path.replace(".xlsx", "_Feature selection.xlsx")
selected_features_df.to_excel(output_file_path, index=False, engine='openpyxl')
