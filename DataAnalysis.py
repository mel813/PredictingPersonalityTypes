from DataPreprocessing import numeric_columns, data
import matplotlib.pyplot as plt
import seaborn as sns

#Visualize the distribution of numeric features to check for outliers
for column in numeric_columns:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x='Personality', y=column, data=data)
    plt.title(f'{column} by Personality Type')
    plt.xlabel('Personality')
    plt.ylabel(column)
    plt.tight_layout()
    plt.show()


#Data visualization using pairplot to see relationships between features
plt.figure(figsize=(10, 8))
sns.pairplot(data, hue='Personality', diag_kind='kde')

# Heatmap displaying correlation between features
plt.figure(figsize=(8, 6))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()


