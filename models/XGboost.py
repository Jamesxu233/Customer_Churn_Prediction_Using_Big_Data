import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
import xgboost as xgb


# Load cleaned dataset
df = pd.read_csv("E:/MSML651 Final Project Data/New Final Project/telco_customer_churn_cleaned.csv")

# Encode target variable
label_encoder = LabelEncoder()
df['Churn'] = label_encoder.fit_transform(df['Churn'])  # 0: No, 1: Yes

# Split features and target
X = df.drop('Churn', axis=1)
y = df['Churn']

# Handle categorical features (One-hot encoding for low-cardinality only)
high_card_cols = [col for col in X.select_dtypes('object').columns if X[col].nunique() > 50]
X = X.drop(columns=high_card_cols)  # Drop high-cardinality features
X = pd.get_dummies(X, drop_first=True)  # One-hot encode remaining categorical features

# Scale numerical features
scaler = StandardScaler()
X[X.select_dtypes(include=np.number).columns] = scaler.fit_transform(X.select_dtypes(include=np.number))

# Final check
print(f"Shape of feature matrix: {X.shape}")
print(f"Target distribution:\n{y.value_counts(normalize=True)}")

# Split the data: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train, y_train)

# Predict
y_pred_xgb = xgb_model.predict(X_test)
print("XGBoost - Classification Report:\n")
print(classification_report(y_test, y_pred_xgb))

# Define hyperparameter distributions
param_dist = {
    'n_estimators': randint(100, 500),
    'max_depth': randint(5, 30),
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 5),
    'max_features': ['sqrt', 'log2']
}

# Randomized Search
random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_dist,
    n_iter=20,
    cv=5,
    scoring='roc_auc',
    random_state=42,
    n_jobs=-1,
    verbose=2
)

# Fit
random_search.fit(X_train, y_train)

print("Best parameters from Randomized Search:\n", random_search.best_params_)
print("Best AUC from CV:", random_search.best_score_)

# Use best estimator from search
best_model = random_search.best_estimator_

# Predict on test set
y_pred_final = best_model.predict(X_test)
y_proba_final = best_model.predict_proba(X_test)[:, 1]

# Evaluation
print("Final Model Evaluation on Test Set:\n")
print(classification_report(y_test, y_pred_final))
print("ROC-AUC Score:", roc_auc_score(y_test, y_proba_final))

# Confusion Matrix
sns.heatmap(confusion_matrix(y_test, y_pred_final), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
