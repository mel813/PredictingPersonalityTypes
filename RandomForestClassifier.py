from DataPreprocessing import data, X, y, X_train, X_test, y_train, y_test, le_target
from DataPreprocessing import get_best_params
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import GridSearchCV

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

# Note the best parameters output may vary depending on the training data and the random state.

# Set up the Random Forest model with the best parameters found
rf_model = RandomForestClassifier(random_state=42, n_estimators=100, min_samples_split=2, min_samples_leaf=4, max_features='sqrt')

# Train on training set and evaluate on test set
rf_model.fit(X_train, y_train)
rf_y_pred = rf_model.predict(X_test)

# Perform 5-fold cross-validation
cv_scores = cross_val_score(rf_model, X, y, cv=5)
print("Cross-val mean accuracy: {:.3f}".format(cv_scores.mean()))
print("Random Forest Model CV Accuracy:", rf_model.score(X_test, y_test))

# Print classification report for Random Forest
print(classification_report(y_test, rf_y_pred, target_names=[str(cls) for cls in le_target.classes_]))

#Visualize Feature importance
importance = rf_model.feature_importances_
feature_names = X_train.columns  # or use whatever list contains your feature names

#Plot the feature importance from Random Forest
plt.figure(figsize=(8, 5))
plt.barh(feature_names, importance)
plt.title("Feature Importance from Random Forest")
plt.ylabel("Importance Score")
plt.tight_layout()
plt.show()