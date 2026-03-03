# =========================================
# House Price Prediction - Random Forest
# =========================================

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Create Dataset
data = {
    "square_footage": [1000, 1500, 2000, 1200, 1800, 2200, 1400, 1700, 2500, 3000],
    "bedrooms": [2, 3, 4, 2, 3, 4, 3, 3, 5, 6],
    "bathrooms": [1, 2, 3, 2, 2, 3, 2, 2, 4, 5],
    "price": [200000, 300000, 400000, 250000, 350000, 450000, 280000, 330000, 500000, 600000]
}

df = pd.DataFrame(data)

# Features and Target
X = df[['square_footage', 'bedrooms', 'bathrooms']]
y = df['price']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Random Forest Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Random Forest Model Output")
print("--------------------------------")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Feature Importance
print("\nFeature Importance:")
for feature, importance in zip(X.columns, model.feature_importances_):
    print(feature, ":", importance)

# Graph
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Random Forest - Actual vs Predicted")
plt.show()