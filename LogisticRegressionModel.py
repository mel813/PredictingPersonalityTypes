from DataPreprocessing import X, y, X_train, X_test, y_train, y_test, le_target
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

#Experimenting with Basic Logistic Regression
lr_model = LogisticRegression()

lr_model.fit(X_train, y_train)
lr_y_pred = lr_model.predict(X_test)

# Perform 5-fold cross-validation
cv_scores = cross_val_score(lr_model, X, y, cv=5)
print("Cross-val mean accuracy: {:.3f}".format(cv_scores.mean()))
print("Logistic Regression Model Accuracy:", lr_model.score(X_test, y_test))
print(classification_report(y_test, lr_y_pred, target_names=[str(cls) for cls in le_target.classes_]))