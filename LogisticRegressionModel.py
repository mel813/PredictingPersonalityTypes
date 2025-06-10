from DataPreprocessing import X_test
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
import numpy as np

'''Testing with Logistic Regression Classifier

This script implements a Logistic Regression classifier to predict personality types based on various features.
It serves as a baseline model for comparison with more complex classifiers.
feature importance visualization is also set up to understand the contribution of each feature in the model's predictions.

'''

#Experimenting with Basic Logistic Regression
def train_logistic_regression(X_train, y_train):
    """
    Function to train a Logistic Regression model.
    
    Parameters:
    X_train: Training features.
    y_train: Training labels.
    
    Returns:
    lr_model: Trained Logistic Regression model.
    lr_y_pred: Predictions on the test set.
    """
    lr_model = LogisticRegression()
    lr_model.fit(X_train, y_train)
    lr_y_pred = lr_model.predict(X_test)
    return lr_model, lr_y_pred

def visualize_feature_importance_logreg(model, feature_names):
    """
    Function to visualize feature importance for a Logistic Regression model.
    This function plots the absolute values of the coefficients of the model as feature importance.
    Parameters:
    model: Trained Logistic Regression model.
    feature_names: List of feature names corresponding to the model's coefficients.
    
    Returns:
    None
    """
    # Get the absolute values of the coefficients and sort them
    importance = np.abs(model.coef_[0])
    indices = np.argsort(importance)[::-1]
    plt.figure(figsize=(8, 5))
    plt.barh(np.array(feature_names)[indices], importance[indices])
    plt.xlabel("Absolute Coefficient Value")
    plt.title("Feature Importance (Logistic Regression)")
    plt.tight_layout()
    plt.show()
