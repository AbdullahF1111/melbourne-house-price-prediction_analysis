## Preparing For Training
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb

# --- 1. Data Preparation and Encoding ---
# Assume df is the cleaned DataFrame from your previous steps.
# A quick check of the data before modeling
print("--- Final DataFrame Info Before Modeling ---")
df.info()

# One-Hot Encode categorical features (The correct approach)
df_encoded = pd.get_dummies(df, columns=['Type', 'Regionname'], drop_first=True)

# Define features (X) and target (y)
X = df_encoded.drop('Price', axis=1)
y = df_encoded['Price']

# Store the list of columns for the prediction function later
training_columns = X.columns.tolist()

# --- 2. Data Splitting ---
# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n--- Data Splitting ---")
print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

# --- 3. Model Training and Evaluation ---
## Training
# We'll use XGBoost, a powerful model often used for this type of data.
print("\n--- Training the XGBoost Regressor Model ---")
model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)
# Calculate the mean price of the target variable
mean_price = df['Price'].mean()
print(f"Mean House Price: {mean_price:,.0f} AUD")
# Evaluation
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n Model Evaluation on Test Data:")
print(f"Root Mean Squared Error (RMSE): {rmse:,.0f} AUD")
print(f"R-squared (R²): {r2:.2f}")
