"""
Task 3: Fraud Detection (Optional)
----------------------------------
This script detects fraudulent transactions using a simple Random Forest model.

Steps:
1. Load dataset from fraud_dataset.csv.
2. Preprocess: handle imbalance with undersampling.
3. Train a Random Forest model.
4. Evaluate accuracy.
5. Predict fraud on new transactions.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Load dataset
df = pd.read_csv("fraud_dataset.csv")
print("Dataset shape:", df.shape)
print(df.head())

# 2. Balance dataset with undersampling
fraud_df = df[df["fraud"] == 1]
nonfraud_df = df[df["fraud"] == 0].sample(len(fraud_df), random_state=42)
balanced_df = pd.concat([fraud_df, nonfraud_df])

X = balanced_df.drop("fraud", axis=1)
y = balanced_df["fraud"]

# 3. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Train Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Evaluate accuracy
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.2f}")

# 5. Predict fraud on new sample transactions
sample_transactions = pd.DataFrame({
    "amount": [20, 500, 5],
    "time": [14, 2, 20],
    "age": [30, 45, 22]
})
preds = model.predict(sample_transactions)

for tx, pred in zip(sample_transactions.to_dict(orient="records"), preds):
    print(f"Transaction: {tx} => Fraudulent: {'Yes' if pred == 1 else 'No'}")
