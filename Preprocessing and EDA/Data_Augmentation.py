import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
import numpy as np


# 1.Basic Data Duplication (for Simple Augmentation)
# Load original Telco Customer Churn dataset
df = pd.read_csv("E:/MSML651 Final Project Data/New Final Project/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Define the desired number of rows (1 million rows in this case)
num_rows_needed = 1000000

# Duplicate the dataset to increase size
df_large = pd.concat([df] * (num_rows_needed // len(df)), ignore_index=True)

# Check the new size of the dataframe
print(df_large.shape)
# Save augmented data to csv
df_large.to_csv('augmented_data.csv', index=False)  # `index=False` to avoid saving row indices