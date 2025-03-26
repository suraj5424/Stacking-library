import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Generate a synthetic regression dataset with 500 samples and 12 features
X, y = make_regression(n_samples=500, n_features=12, noise=0.1, random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define stacking layers
# Define models for each layer (for regression)
layer1 = [LinearRegression(), RandomForestRegressor(n_estimators=100, random_state=42)]
layer2 = [
    RandomForestRegressor(n_estimators=50, random_state=42), 
    LinearRegression(), 
    RandomForestRegressor(n_estimators=200, random_state=42)
]
layer3 = [XGBRegressor(random_state=42), XGBRegressor(random_state=42)]

# Define the meta model
meta_model = LinearRegression()

# Combine layers into a list
layers = [layer1, 
          layer2, layer3
         ]

# Initialize and train Stacking Model
model = StackingEnsemble(layers= layers, meta_model=meta_model)
model.fit(X_train, y_train)

# Print the structure of the stacking model
model.print_structure()

# Make predictions
predictions = model.predict(X_test)
print("Predictions: ", predictions[:10])
