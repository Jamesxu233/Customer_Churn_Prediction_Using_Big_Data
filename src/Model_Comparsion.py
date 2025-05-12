import matplotlib.pyplot as plt
import pandas as pd

# Data from the three model evaluations
data = {
    'Model': ['Logistic Regression', 'Random Forest (RndSearch)', 'XGBoost'],
    'Precision (Churn=1)': [0.66, 1.00, 0.99],
    'Precision (Churn=0)': [0.85, 1.00, 1.00],
    'Recall (Churn=1)': [0.55, 0.99, 0.99],
    'Recall (Churn=0)': [0.90, 1.00, 1.00],
    'F1-Score': [0.80, 1.00, 1.00],
    'Accuracy': [0.81, 1.00, 1.00],
    'ROC-AUC': [0.8474, 0.99998, 0.99995]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Plotting
fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

# Classification metrics
df.set_index('Model')[['Precision (Churn=1)', 'Precision (Churn=0)', 'Recall (Churn=1)', 'Recall (Churn=0)', 'F1-Score', 'Accuracy']].plot(kind='bar', ax=axs[0])
axs[0].set_title('Model Comparison - Classification Metrics')
axs[0].set_ylabel('Score')
axs[0].legend(loc='lower right')
axs[0].set_ylim(0.5, 1.05)
axs[0].grid(True, linestyle='--', alpha=0.6)

# ROC-AUC Score
df.set_index('Model')[['ROC-AUC']].plot(kind='bar', ax=axs[1], color='orange')
axs[1].set_title('Model Comparison - ROC-AUC')
axs[1].set_ylabel('AUC')
axs[1].set_ylim(0.5, 1.05)
axs[1].grid(True, linestyle='--', alpha=0.6)

plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
