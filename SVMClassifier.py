from DataPreprocessing import X, y, X_train, X_test, y_train, y_test, le_target
from DataPreprocessing import get_best_params
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance

'''Testing with SVM Classifiers

This script trains a Support Vector Machine (SVM) classifier to predict personality types based on various features.
It includes hyperparameter tuning using GridSearchCV and evaluates the model's performance using cross-validation and classification reports.
Feature importance visualization is also set up using permutation importance to understand the contribution of each feature in the model's predictions.

'''

#Set up the SVM model with initial parameters
svm_model=SVC(kernel="rbf", C=1, gamma=1)

# Define the range of hyperparameters for tuning
C_range = [2**i for i in range(-5, 6)]
gamma_range = [2**i for i in range(-5, 6)]

# Define parameter grid
param_grid = {
    'C': C_range,
    'gamma': gamma_range
}

# Setup GridSearch with cross-validation
(best_params, best_score) = get_best_params(svm_model, param_grid, X_train, y_train)

print("Best combination of parameters:", best_params)
print("Best cross-validation accuracy:", best_score)

def train_svm(X_train, y_train):
    """
    Function to train a Support Vector Machine (SVM) model.
    
    Parameters:
    X_train: Training features.
    y_train: Training labels.
    
    Returns:
    svm_model: Trained SVM model.
    svm_y_pred: Predictions on the test set.
    """
    
    # Train the SVM model with the best parameters
    svm_model = SVC(**best_params)
    
    # Fit the model on the training data
    svm_model.fit(X_train, y_train)
    
    # Predict on the test set
    svm_y_pred = svm_model.predict(X_test)
    
    return svm_model, svm_y_pred


def visualize_feature_importance_svm(svm_model, X_train):
    """
    Function to visualize feature importance using permutation importance.
    
    Parameters:
    svm_model: Trained SVM model.
    X_train: Training features.
    
    Returns:
    None
    """
    
    # Calculate permutation importance
    result = permutation_importance(svm_model, X_train, y_train, n_repeats=10, random_state=42)
    
    importances = result.importances_mean
    feature_names = X_train.columns if hasattr(X_train, 'columns') else [f"Feature {i}" for i in range(X_train.shape[1])]
    
    # Plot the feature importance
    plt.figure(figsize=(20, 6))
    plt.barh(feature_names, importances)
    plt.xlabel("Mean Importance")
    plt.title("Permutation Feature Importance (SVM)")
    plt.show()
