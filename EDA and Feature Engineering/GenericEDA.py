# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load your dataset
df = pd.read_csv('your_dataset.csv')  # Replace with your actual file

# ================================
# Basic Dataset Information
# ================================
print("First 5 Rows:\n", df.head())
print("\nShape of Dataset:", df.shape)
print("\nData Types and Non-Null Counts:")
df.info()
print("\nStatistical Summary:\n", df.describe())

# Check for missing values
print("\nMissing Values:\n", df.isnull().sum())

# Check for duplicated rows
print("\nDuplicated Rows:", df.duplicated().sum())

# ================================
# Target/Label Distribution (if any)
# ================================
target_column = 'target'  # Change to your actual target/label column
if target_column in df.columns:
    print("\nTarget Distribution:\n", df[target_column].value_counts())
    print("\nTarget Distribution (%):\n", df[target_column].value_counts(normalize=True) * 100)

    sns.countplot(x=target_column, data=df)
    plt.title('Target Class Distribution')
    plt.show()

# ================================
# Correlation Heatmap
# ================================
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), cmap='coolwarm', annot=False, linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

# ================================
# Distribution Plots for Numerical Features
# ================================
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

for col in numerical_cols:
    plt.figure()
    sns.histplot(df[col], bins=50, kde=True)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()

# ================================
# Boxplots for Numerical Features vs Target (if applicable)
# ================================
if target_column in df.columns:
    for col in numerical_cols:
        if col != target_column:
            plt.figure()
            sns.boxplot(x=target_column, y=col, data=df)
            plt.title(f'{col} vs {target_column}')
            plt.show()

# ================================
# Countplots for Categorical Features
# ================================
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

for col in categorical_cols:
    plt.figure()
    sns.countplot(x=col, data=df)
    plt.title(f'Distribution of {col}')
    plt.xticks(rotation=45)
    plt.show()
