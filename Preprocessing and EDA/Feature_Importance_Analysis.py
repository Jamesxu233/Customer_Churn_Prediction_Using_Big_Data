import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import shap  # For model interpretability
from sklearn.inspection import permutation_importance

# Load cleaned dataset
df = pd.read_csv("E:/MSML651 Final Project Data/New Final Project/telco_customer_churn_cleaned.csv")

# Encode binary target
label_encoder = LabelEncoder()
df['Churn'] = label_encoder.fit_transform(df['Churn'])

# Separate features and target
X = df.drop(columns=['Churn'])
y = df['Churn']

# Check cardinality of categorical columns
for col in X.select_dtypes(include='object').columns:
    print(f"{col}: {X[col].nunique()} unique values")

# Drop high-cardinality columns that don't help
X = X.drop(columns=['customerID'])  # or whatever column has huge cardinality

# One-hot encode categorical variables
X = pd.get_dummies(X, drop_first=True)

# Standardize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Fit a Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Feature importance from model
importances = rf_model.feature_importances_
feature_names = X.columns

# Create a DataFrame for visualization
rf_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
rf_importance_df = rf_importance_df.sort_values(by='Importance', ascending=False)

# Plot top 20 features
plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=rf_importance_df.head(20))
plt.title("Top 20 Feature Importances from Random Forest")
plt.show()
