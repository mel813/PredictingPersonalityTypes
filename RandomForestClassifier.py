from DataPreprocessing import data, X, y, X_train, X_test, y_train, y_test, le_target
from DataPreprocessing import get_best_params
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import GridSearchCV

'''Testing with Random Forest Classifiers

This script trains a Random Forest(RF) classifier to predict personality types based on various features.
It includes hyperparameter tuning using GridSearchCV and evaluates the model's performance using cross-validation and classification reports.
Feature importance visualization is also set up to understand the contribution of each feature in the model's predictions.

'''

#Experimenting with Random Forest Classifier
rf = RandomForestClassifier(random_state=42)

#Define the parameter grid to search
param_grid = {
    'n_estimators': [100, 300, 500],
    'max_depth': [None, 8, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

# Setup GridSearch with cross-validation
(best_params, best_score) = get_best_params(rf, param_grid, X_train, y_train)

# Best parameters and best score
print("Best parameters:", best_params)
print("Best cross-validation accuracy:", best_score)

# Note the best parameters output may vary depending on the training data split and the random state.

def train_random_forest(X_train, y_train):
    """
    Function to train a Random Forest model.
    
    Parameters:
    X_train: Training features.
    y_train: Training labels.
    
    Returns:
    rf_model: Trained Random Forest model.
    rf_y_pred: Predictions on the test set.
    """

    # Train the Random Forest model with the best parameters
    rf_model = RandomForestClassifier(**best_params, random_state=42)

    # Fit the model on the training data
    rf_model.fit(X_train, y_train)

    # Predict on the test set
    rf_y_pred = rf_model.predict(X_test)

    # Return the trained model and predictions
    return rf_model, rf_y_pred

def visualize_feature_importance_rf(rf_model, X_train):
    """
    Function to visualize feature importance from the Random Forest model.
    
    Parameters:
    rf_model: Trained Random Forest model.
    X_train: Training features.
    
    Returns:
    None
    """
    importance = rf_model.feature_importances_
    feature_names = X_train.columns 

    # Plot the feature importance
    plt.figure(figsize=(8, 5))
    plt.barh(feature_names, importance)
    plt.title("Feature Importance from Random Forest")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.show()
