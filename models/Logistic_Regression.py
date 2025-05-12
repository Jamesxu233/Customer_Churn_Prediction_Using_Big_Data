import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# Load cleaned dataset
df = pd.read_csv("E:/MSML651 Final Project Data/New Final Project/telco_customer_churn_cleaned.csv")

# Encode target variable
label_encoder = LabelEncoder()
df['Churn'] = label_encoder.fit_transform(df['Churn'])  # 0: No, 1: Yes

# Separate features and target
X = df.drop('Churn', axis=1)
y = df['Churn']

# Drop high-cardinality categorical features (if any)
high_card_cols = [col for col in X.select_dtypes('object').columns if X[col].nunique() > 50]
X = X.drop(columns=high_card_cols)

# One-hot encode remaining categorical variables
X = pd.get_dummies(X, drop_first=True)

# Scale numerical features
scaler = StandardScaler()
X[X.select_dtypes(include=np.number).columns] = scaler.fit_transform(X.select_dtypes(include=np.number))

# Split the dataset into train and test sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train samples: {X_train.shape}, Test samples: {X_test.shape}")

# Initialize and train logistic regression
logreg = LogisticRegression(random_state=42, max_iter=1000)  # increase max_iter if needed
logreg.fit(X_train, y_train)

# Predict
y_pred = logreg.predict(X_test)
y_prob = logreg.predict_proba(X_test)[:, 1]

# Evaluation
print("Logistic Regression - Classification Report:")
print(classification_report(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_prob))

# Confusion Matrix
import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Define parameter grid
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],     # Inverse of regularization strength
    'penalty': ['l1', 'l2'],          # Regularization type
    'solver': ['liblinear']           # Solver that supports both l1 and l2
}

# Initialize logistic regression
logreg = LogisticRegression(max_iter=1000, random_state=42)

# Grid search with 5-fold cross-validation
grid_search = GridSearchCV(
    estimator=logreg,
    param_grid=param_grid,
    scoring='roc_auc',
    cv=5,
    n_jobs=-1,
    verbose=2
)

# Fit
grid_search.fit(X_train, y_train)

# Best model
best_logreg = grid_search.best_estimator_

# Evaluation
y_pred_best = best_logreg.predict(X_test)
y_prob_best = best_logreg.predict_proba(X_test)[:, 1]

print("Best Logistic Regression Parameters:", grid_search.best_params_)
print("Test ROC-AUC Score:", roc_auc_score(y_test, y_prob_best))
print("Classification Report:")
print(classification_report(y_test, y_pred_best))

# Get feature importance from coefficients
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': best_logreg.coef_[0]
}).sort_values(by='Coefficient', key=abs, ascending=False)

print(importance_df.head(10))
