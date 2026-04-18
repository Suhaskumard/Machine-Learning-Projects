from sklearn.neural_network import MLPClassifier


def build_model():
    return MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        max_iter=50,
        early_stopping=True,
        n_iter_no_change=5,
        random_state=42,
        verbose=True,
    )
