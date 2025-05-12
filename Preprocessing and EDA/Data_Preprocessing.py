import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from scipy import stats

# Load dataset
df = pd.read_csv("E:/MSML651 Final Project Data/New Final Project/augmented_data.csv")

# Display basic information about the dataset
print(df.info())
print(df.head())

# Check for missing values
print(df.isnull().sum())

# Check for non-numeric values and empty strings in the numerical columns
numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
for col in numerical_features:
    print(f"Non-numeric values in {col}:\n", df[col][~df[col].apply(lambda x: isinstance(x, (int, float)))])
    print(f"Empty strings in {col}:\n", df[col][df[col] == ''])

# Replace empty strings with NaN
df[numerical_features] = df[numerical_features].replace('', float('nan'))

# Check if there are still missing values after the replacement
print(df.isnull().sum())

# Impute missing values (e.g., with the column mean)
df[numerical_features] = df[numerical_features].fillna(df[numerical_features].mean())

# Convert columns to numeric (forcing errors to NaN)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Check for any NaN values after conversion
print(df['TotalCharges'].isnull().sum())

# Impute missing values (optional)
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].mean())

# Initialize label encoder
label_encoder = LabelEncoder()

# Encode 'Churn' as 0 (No) and 1 (Yes)
df['Churn'] = label_encoder.fit_transform(df['Churn'])

# Encode other binary categorical variables
df['SeniorCitizen'] = label_encoder.fit_transform(df['SeniorCitizen'])

# View the dataset after encoding
print(df.info())

# List of numerical features
numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']

# Initialize the StandardScaler
scaler = StandardScaler()

# Apply scaling to the numerical features
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# Initialize the MinMaxScaler
min_max_scaler = MinMaxScaler()

# Apply Min-Max scaling to the numerical features
df[numerical_features] = min_max_scaler.fit_transform(df[numerical_features])

# Calculate Z-scores for numerical columns
z_scores = stats.zscore(df[numerical_features])

# Identify outliers based on Z-score (|Z| > 3)
outliers = (z_scores > 3) | (z_scores < -3)

# Remove rows with outliers
df_no_outliers = df[~outliers.any(axis=1)]

# Calculate Q1 (25th percentile) and Q3 (75th percentile)
Q1 = df[numerical_features].quantile(0.25)
Q3 = df[numerical_features].quantile(0.75)

# Calculate IQR
IQR = Q3 - Q1

# Filter the data to remove outliers (values outside 1.5 * IQR)
df_no_outliers = df[~((df[numerical_features] < (Q1 - 1.5 * IQR)) | (df[numerical_features] > (Q3 + 1.5 * IQR))).any(axis=1)]

# Convert 'TotalCharges' to numeric, coercing errors to NaN
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Check data types
print(df.dtypes)

# Verify the cleaned data
print(df.info())

# Save the cleaned dataset
df.to_csv("telco_customer_churn_cleaned.csv", index=False)