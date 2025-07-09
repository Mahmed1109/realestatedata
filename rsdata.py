import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer


# Step 2: Load dataset (California Housing)
url = 'https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv'
df = pd.read_csv(url)


print("Initial data shape:", df.shape)
print(df.head())

df['rooms_per_household'] = df['total_rooms'] / df['households']
df['bedrooms_per_room'] = df['total_bedrooms'] / df['total_rooms']
df['population_per_household'] = df['population'] / df['households']

print("\nMissing values before imputation:")
print(df.isnull().sum())

imputer = SimpleImputer(strategy='median')
df['total_bedrooms'] = imputer.fit_transform(df[['total_bedrooms']])

print("\nMissing values after imputation:")
print(df.isnull().sum())

imputer = SimpleImputer(strategy='median')
df['total_bedrooms'] = imputer.fit_transform(df[['total_bedrooms']])

print("\nMissing values after imputation:")
print(df.isnull().sum())

#Visualise with Plotly 

fig1 = px.histogram(df, x='median_house_value', nbins=50, title='Distribution of Median House Value')
fig1.show()

fig2 = px.scatter_mapbox(
    df, lat="latitude", lon="longitude", color="median_house_value", size='population',
    color_continuous_scale=px.colors.cyclical.IceFire, size_max=15, zoom=8,
    mapbox_style="carto-positron", title="Geographic Distribution of House Prices"
)
fig2.show()
features = [
    'median_income', 'housing_median_age', 'total_rooms', 'total_bedrooms',
    'population', 'households', 'rooms_per_household', 'bedrooms_per_room', 'population_per_household',
    'latitude', 'longitude'
]
X = df[features]
y = df['median_house_value']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
print("\nLinear Regression RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_lr)))
print("Linear Regression R2:", r2_score(y_test, y_pred_lr))

rf = RandomForestRegressor(random_state=42)
params_rf = {
    'n_estimators': [50, 100],
    'max_features': ['auto', 'sqrt'],
    'max_depth': [10, 20, None]
}
grid_rf = GridSearchCV(rf, param_grid=params_rf, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
grid_rf.fit(X_train, y_train)

print("\nBest Random Forest params:", grid_rf.best_params_)

best_rf = grid_rf.best_estimator_
y_pred_rf = best_rf.predict(X_test)
print("Random Forest RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_rf)))
print("Random Forest R2:", r2_score(y_test, y_pred_rf))

gbr = GradientBoostingRegressor(random_state=42)
params_gbr = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 5]
}
grid_gbr = GridSearchCV(gbr, param_grid=params_gbr, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
grid_gbr.fit(X_train, y_train)

print("\nBest Gradient Boosting params:", grid_gbr.best_params_)

best_gbr = grid_gbr.best_estimator_
y_pred_gbr = best_gbr.predict(X_test)
print("Gradient Boosting RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_gbr)))
print("Gradient Boosting R2:", r2_score(y_test, y_pred_gbr))