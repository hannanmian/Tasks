# Data Science Internship Tasks

This repository contains the solutions to the internship tasks as described in the assignment PDF.  
Each task is implemented as a separate notebook or Python script.

---

## ✅ Task 1: EDA and Visualization
- **File:** `Task1_EDA_Titanic.ipynb`
- **Dataset:** Titanic (`train.csv`)
- **Steps:**
  - Loaded dataset with Pandas.
  - Cleaned data (filled missing values, removed duplicates).
  - Visualizations:
    - Bar chart of survival counts.
    - Histogram of age distribution.
    - Correlation heatmap.
  - **Outcome:** Notebook with EDA process, visualizations, and insights.

---

## ✅ Task 2: Sentiment Analysis
- **File:** `Task2_Sentiment.py` (can also be converted to Jupyter Notebook if required)
- **Dataset:** `imdb_1000_reviews.csv` (1000 reviews: 500 positive, 500 negative)
- **Steps:**
  - Preprocessed reviews (lowercase, removed special characters/stopwords).
  - Converted text into numerical format using **CountVectorizer**.
  - Trained a **Logistic Regression** model.
  - Evaluated accuracy and predicted sentiment for new reviews.
- **Outcome:** Python script that classifies reviews as positive or negative.

---

## ✅ Task 3: Fraud Detection (Optional)
- **File:** `Task3_Fraud.py`
- **Dataset:** `fraud_dataset.csv` (1000 transactions, ~5% fraud cases)
- **Steps:**
  - Preprocessed data (handled imbalance using undersampling).
  - Trained a **Random Forest** model.
  - Evaluated model accuracy.
  - Predicted fraud for new sample transactions.
- **Outcome:** Python script that predicts whether a transaction is fraudulent or not.

---

## ✅ Task 4: Predicting House Prices
- **File:** `Task4_HousePrices.py`
- **Dataset:** `housing.csv` (Boston Housing Dataset)
- **Steps:**
  - Normalized numerical features with **StandardScaler**.
  - Trained a **Linear Regression** model.
  - Evaluated performance with **RMSE**.
  - Predicted price for a sample house.
- **Outcome:** Python script that predicts house prices and shows RMSE.

---

## 🚀 How to Run

1. Clone this repository:
   ```bash
   git clone <your-repo-link>
   cd <your-repo-folder>
# Tasks