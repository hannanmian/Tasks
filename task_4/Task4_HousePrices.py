import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# 1. Load dataset with column names
column_names = [
    "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE",
    "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"
]
df = pd.read_csv("housing.csv", delim_whitespace=True, names=column_names)

print("Dataset shape:", df.shape)
print(df.head())

# 2. Features and target
X = df.drop("MEDV", axis=1)
y = df["MEDV"]

# 3. Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Train Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Evaluate model
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

# Predict price for first row
sample_house = X.iloc[0].values.reshape(1, -1)
sample_scaled = scaler.transform(sample_house)
predicted_price = model.predict(sample_scaled)[0]
print(f"Predicted House Price for first row: ${predicted_price*1000:.2f}")
