import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("E:/MSML651 Final Project Data/New Final Project/telco_customer_churn_cleaned.csv")

# Display basic information about the dataset
print(df.info())  # Check the types of columns and missing values
print(df.head())  # Preview the first few rows of the dataset

# Get summary statistics for numerical columns
print(df.describe())  # Includes mean, std, min, 25%, 50%, 75%, max

# Get summary statistics for categorical columns
print(df['Churn'].value_counts())  # Check distribution of target variable
print(df['PaymentMethod'].value_counts())  # Check distribution of payment methods

# Visualize the target variable distribution (Churn)
sns.countplot(data=df, x='Churn')
plt.title("Churn Distribution")
plt.show()

# Visualize the distribution of tenure
sns.histplot(df['tenure'], kde=True, color='blue')
plt.title('Distribution of Tenure')
plt.show()

# Visualize the distribution of MonthlyCharges
sns.histplot(df['MonthlyCharges'], kde=True, color='green')
plt.title('Distribution of Monthly Charges')
plt.show()

# Visualize the distribution of TotalCharges
sns.histplot(df['TotalCharges'], kde=True, color='red')
plt.title('Distribution of Total Charges')
plt.show()

# Calculate correlation matrix for numerical features
corr_matrix = df.corr()

# Visualize correlation matrix using heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()

# Visualize churn distribution by payment method
sns.countplot(data=df, x='PaymentMethod', hue='Churn')
plt.title('Churn Distribution by Payment Method')
plt.show()

# Visualize churn distribution by contract type
sns.countplot(data=df, x='Contract', hue='Churn')
plt.title('Churn Distribution by Contract Type')
plt.show()

# Visualize churn distribution by internet service type
sns.countplot(data=df, x='InternetService', hue='Churn')
plt.title('Churn Distribution by Internet Service')
plt.show()

# Feature Engineering
# Create a new feature "is_high_spender" based on MonthlyCharges
df['is_high_spender'] = df['MonthlyCharges'] > 100

# Show the new feature's distribution
sns.countplot(data=df, x='is_high_spender', hue='Churn')
plt.title('Churn Distribution by High Spender')
plt.show()

# Create a new categorical feature based on tenure
df['tenure_category'] = pd.cut(df['tenure'], bins=[0, 12, 24, 48, 72, 84], 
                               labels=['0-1 year', '1-2 years', '2-4 years', '4-6 years', '6+ years'])

# Show churn distribution by tenure category
sns.countplot(data=df, x='tenure_category', hue='Churn')
plt.title('Churn Distribution by Tenure Category')
plt.show()

# Boxplot for MonthlyCharges vs Churn
sns.boxplot(x='Churn', y='MonthlyCharges', data=df)
plt.title('Monthly Charges vs Churn')
plt.show()

# Boxplot for TotalCharges vs Churn
sns.boxplot(x='Churn', y='TotalCharges', data=df)
plt.title('Total Charges vs Churn')
plt.show()

# Bar plot for Contract vs Churn
sns.countplot(data=df, x='Contract', hue='Churn')
plt.title('Contract vs Churn')
plt.show()

# Bar plot for InternetService vs Churn
sns.countplot(data=df, x='InternetService', hue='Churn')
plt.title('Internet Service vs Churn')
plt.show()