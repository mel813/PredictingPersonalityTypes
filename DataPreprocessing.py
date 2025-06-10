import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, auc
from sklearn.impute import SimpleImputer 

'''

Data Preprocessing and Model Training for Personality Prediction
This script performs data preprocessing on a personality dataset, which is used to train various machine learning models (Logistic Regression, Random Forest, SVM).
It also includes a function for hyperparameter tuning for the models using GridSearchCV.
It includes accounting for missing values, encoding categorical variables.

'''

def get_best_params(model, param_grid, X_train, y_train):
    """
    Function to perform grid search for hyperparameter tuning.
    
    Parameters:
    model: The machine learning model to tune.
    param_grid: Dictionary with parameters names as keys and lists of parameter settings to try as values.
    X_train: Training features.
    y_train: Training labels.
    
    Returns:
    best_params: Dictionary with the best parameters found.
    best_score: The best score achieved with the best parameters.
    """
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, scoring='accuracy')

    #Fit on the training data
    grid_search.fit(X_train, y_train)

    # Return the best parameters and the best score
    return grid_search.best_params_, grid_search.best_score_

# Read personality dataset from csv
data = pd.read_csv('data/personality_dataset.csv')

#Visualize Data Set Info
print("Data Shape:", data.shape)  # Check the shape of the dataset
print("\nData Info:")
data.info()

print(data.head()) # Display the first few rows of the dataset

#Adjusting for null values
numeric_columns = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']

print("\nMissing Values Per Feature:")
print(data.isnull().sum())

#Fill missing values with the mean for numeric columns
imputer = SimpleImputer(strategy='mean')
data[numeric_columns] = imputer.fit_transform(data[numeric_columns])

# Encode categorical variables(Stage_fear and Drained_after_socializing)
le = LabelEncoder()
for col in ['Stage_fear', 'Drained_after_socializing']:
    data[col] = le.fit_transform(data[col])

# Encode the target variable
le_target = LabelEncoder()
data['Personality'] = le_target.fit_transform(data['Personality'])

### Clean data before visualization and analysis ###

# Set up target variable and features
# Note: The target variable is 'Personality' and the features are all other columns

# Tested dropping stage_fear from the dataset due to high correlation with Drained_after_socializing, 
# did not consistently improve model performance
X = data.drop(['Personality'], axis=1)
y = data['Personality']

#Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
