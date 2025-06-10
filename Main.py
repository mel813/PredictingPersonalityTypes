from DataPreprocessing import X, y, X_train, X_test, y_train, y_test, le_target
from LogisticRegressionModel import train_logistic_regression, visualize_feature_importance_logreg
from RandomForestClassifier import train_random_forest, visualize_feature_importance_rf
from SVMClassifier import train_svm, visualize_feature_importance_svm
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, roc_curve, auc
import matplotlib.pyplot as plt

'''

Main script to compare different machine learning models for personality prediction.
This script handles calling the training functions from each model type and predicting on them.
It evaluates the models' performance using accuracy, cross-validation scores, and classification reports.
This script also visualizes feature importance for each model and plots the ROC curve for comparison.

'''

#Set up the models to be trained
#Compares different machine learning models for personality prediction.
models = {
    'Logistic Regression': {
        'model': train_logistic_regression,
        'proba_func': lambda m: m.predict_proba(X_test)[:, 1],
        'feature_importance': visualize_feature_importance_logreg,
        'color': 'darkorange'
    },
    'Random Forest': {
        'model': train_random_forest,
        'proba_func': lambda m: m.predict_proba(X_test)[:, 1],
        'feature_importance': visualize_feature_importance_rf,
        'color': 'blue'
    },
    'SVM': {
        'model': train_svm,
        'proba_func': lambda m: m.decision_function(X_test),
        'feature_importance': visualize_feature_importance_svm,
        'color': 'green'
    }
}

def evaluate_model(name, model, y_pred):
    '''Function to evaluate the model's performance.
    This function prints the accuracy of the model and a classification report.

    Parameters:
    name: Name of the model.
    model: The trained model.
    y_pred: Predictions made by the model.

    Returns:
    None
    '''

    print(f"{name} Model Accuracy: {model.score(X_test, y_test):.3f}")
    cv_scores = cross_val_score(model, X, y, cv=5)
    print("Cross-val mean accuracy: {:.3f}".format(cv_scores.mean()))
    print(classification_report(y_test, y_pred, target_names=[str(cls) for cls in le_target.classes_]))

results = {}

# Train and evaluate each model in the models dictionary. Perform cross-validation and print classification reports.
for name, info in models.items():
    model_func = info['model']
    model, y_pred = model_func(X_train, y_train)
    results[name] = (model, y_pred)
    print('evaluating', name)
    evaluate_model(name, model, y_pred)   

# Visualize feature importance for Logistic Regression
if 'Logistic Regression' in results:
    lr_model, _ = results['Logistic Regression']
    visualize_feature_importance_logreg(lr_model, X_train.columns)

# Visualize feature importance for Random Forest
if 'Random Forest' in results:
    rf_model, _ = results['Random Forest']
    visualize_feature_importance_rf(rf_model, X_train)

# Visualize feature importance for SVM
if 'SVM' in results:
    svm_model, _ = results['SVM']
    visualize_feature_importance_svm(svm_model, X_train)

#Compare model accuracies and plot ROC curve
plt.figure(figsize=(8, 6))
for name, info in models.items():
    model = results[name][0]

    # Compare model accuracy scores
    scores = cross_val_score(model, X, y, cv=5)
    print(f"{name} CV Accuracy: {scores.mean():.4f}")
    y_score = info['proba_func'](model)
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=info['color'], lw=2, label=f'{name} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random guess')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()