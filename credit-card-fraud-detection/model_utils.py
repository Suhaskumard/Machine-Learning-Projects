# model_utils.py

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def train_model(X_train, Y_train, max_iter):
    model = LogisticRegression(max_iter=max_iter)
    model.fit(X_train, Y_train)
    return model


def evaluate_model(model, X_test, Y_test):
    Y_pred = model.predict(X_test)

    return {
        "Accuracy": accuracy_score(Y_test, Y_pred),
        "Precision": precision_score(Y_test, Y_pred),
        "Recall": recall_score(Y_test, Y_pred),
        "F1 Score": f1_score(Y_test, Y_pred)
    }
