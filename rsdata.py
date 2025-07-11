# Step 1: Import libraries
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

import warnings
warnings.filterwarnings('ignore')

# Step 2: Load dataset
url = 'https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv'
df = pd.read_csv(url)

# Step 3: Feature engineering
df['rooms_per_household'] = df['total_rooms'] / df['households']
df['bedrooms_per_room'] = df['total_bedrooms'] / df['total_rooms']
df['population_per_household'] = df['population'] / df['households']

# Step 4: Define feature set
features = [
    'median_income', 'housing_median_age', 'total_rooms', 'total_bedrooms',
    'population', 'households', 'rooms_per_household', 'bedrooms_per_room',
    'population_per_household', 'latitude', 'longitude'
]

# Step 5: Impute all missing values in the features
imputer = SimpleImputer(strategy='median')
df[features] = imputer.fit_transform(df[features])

# Optional sanity check for missing values
assert df[features].isnull().sum().sum() == 0, "Missing values still exist!"

# Step 6: Visualizations
px.histogram(df, x='median_house_value', nbins=50, title='Distribution of Median House Value').show()

px.scatter_mapbox(
    df, lat="latitude", lon="longitude", color="median_house_value", size='population',
    color_continuous_scale=px.colors.cyclical.IceFire, size_max=15, zoom=8,
    mapbox_style="carto-positron", title="Geographic Distribution of House Prices"
).show()

# Step 7: Train-test split
X = df[features]
y = df['median_house_value']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 8: Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
print("\nLinear Regression RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_lr)))
print("Linear Regression R2:", r2_score(y_test, y_pred_lr))

# Step 9: Random Forest with Grid Search
rf = RandomForestRegressor(random_state=42)
params_rf = {
    'n_estimators': [50, 100],
    'max_depth': [10, 20, None],
    'max_features': ['auto', 'sqrt']
}
grid_rf = GridSearchCV(rf, param_grid=params_rf, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
grid_rf.fit(X_train, y_train)
best_rf = grid_rf.best_estimator_
y_pred_rf = best_rf.predict(X_test)
print("\nRandom Forest RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_rf)))
print("Random Forest R2:", r2_score(y_test, y_pred_rf))
print("Best RF Params:", grid_rf.best_params_)

# Step 10: Gradient Boosting
gbr = GradientBoostingRegressor(random_state=42)
params_gbr = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 5]
}
grid_gbr = GridSearchCV(gbr, param_grid=params_gbr, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
grid_gbr.fit(X_train, y_train)
best_gbr = grid_gbr.best_estimator_
y_pred_gbr = best_gbr.predict(X_test)
print("\nGradient Boosting RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_gbr)))
print("Gradient Boosting R2:", r2_score(y_test, y_pred_gbr))
print("Best GBR Params:", grid_gbr.best_params_)

# Step 11: Feature Importance
feat_imp = pd.DataFrame({
    'feature': features,
    'importance': best_rf.feature_importances_
}).sort_values(by='importance', ascending=False)

px.bar(feat_imp, x='feature', y='importance', title='Feature Importance (Random Forest)').show()

# Step 12: Summary Report
print("\n--- SUMMARY REPORT ---")
print(f"Dataset shape: {df.shape}")
print(f"Features used: {features}")
print(f"\nLinear Regression -> RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_lr)):.2f}, R2: {r2_score(y_test, y_pred_lr):.2f}")
print(f"Random Forest    -> RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_rf)):.2f}, R2: {r2_score(y_test, y_pred_rf):.2f}")
print(f"Gradient Boost   -> RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_gbr)):.2f}, R2: {r2_score(y_test, y_pred_gbr):.2f}")
print("\nTop 5 Important Features (RF):")
print(feat_imp.head())
