from DataPreprocessing import numeric_columns, data
import matplotlib.pyplot as plt
import seaborn as sns

'''
Data Visualization for Personality Dataset
This script visualizes the distribution of numeric features in the personality dataset to check for outliers and relationships between features.
It includes box plots for each numeric feature grouped by personality type, a pairplot to visualize relationships, and a heatmap to display feature correlations.

'''

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
sns.pairplot(data, hue='Personality', diag_kind='kde', height=1.5)

# Heatmap displaying correlation between features
plt.figure(figsize=(8, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()


