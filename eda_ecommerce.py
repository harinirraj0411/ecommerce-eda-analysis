import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your CSV dataset with encoding fix
df = pd.read_csv("data.csv", encoding="latin1", encoding_errors="ignore")
print("Dataset loaded successfully!")
print(df.head())

# Basic info
print("\n===== Basic Info =====")
print(df.info())

print("\n===== Missing Values =====")
print(df.isnull().sum())

print("\n===== Statistical Summary =====")
print(df.describe())

# Correlation heatmap (numeric only)
numeric_df = df.select_dtypes(include=['int64', 'float64'])

plt.figure(figsize=(8, 5))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()


# Distribution of each numeric column
df.hist(figsize=(12,10))
plt.suptitle("Feature Distributions")
plt.show()
