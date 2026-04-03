import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset (Iris)
data = load_iris()
X = data.data
y = data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = DecisionTreeClassifier(max_depth=3)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Visualize tree
plt.figure(figsize=(10, 6))
plot_tree(model, feature_names=data.feature_names,
          class_names=data.target_names, filled=True)
plt.title("Decision Tree")
plt.show()

# Simple prediction
sample = X_test[0]
prediction = model.predict([sample])
print("Sample Prediction:", data.target_names[prediction][0])