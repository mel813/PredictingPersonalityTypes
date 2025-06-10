from DataPreprocessing import X, y, X_train, X_test, y_train, y_test, le_target
from DataPreprocessing import get_best_params
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.svm import SVC

#Experimenting with SVM Classifiers
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

# Use the best estimator to predict
# Retrain the SVM model with the best parameters
svm_model = SVC(C=0.03125, gamma=0.03125)
svm_model.fit(X_train, y_train)
svm_y_pred = svm_model.predict(X_test)
svm_scores = cross_val_score(svm_model, X, y, cv=5)
print("cross-val mean-accuracy: {:.3f}".format(np.mean(svm_scores)))
print("SVM Model CV Accuracy:", svm_model.score(X_test, y_test))
print(classification_report(y_test, svm_y_pred, target_names=[str(cls) for cls in le_target.classes_]))

from sklearn.inspection import permutation_importance
result = permutation_importance(svm_model, X_test, y_test, n_repeats=10, random_state=42)

importances = result.importances_mean
feature_names = X.columns if hasattr(X, 'columns') else [f"Feature {i}" for i in range(X.shape[1])]

plt.barh(feature_names, importances)
plt.xlabel("Mean Importance")
plt.title("Permutation Feature Importance (SVM)")
plt.show()
