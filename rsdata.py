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