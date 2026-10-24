# Import the tools we need from scikit-learn.
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


print("Beginner's Guide to Binary Classification with an ANN")
print("Using Scikit-learn's Make Moons Dataset")
print("=" * 58)

# 1. Make sample data. Each row has two measurements and belongs to class 0 or 1.
features, labels = make_moons(n_samples=500, noise=0.2, random_state=42)

# 2. Split the data: use most rows for learning and the rest for checking.
train_features, test_features, train_labels, test_labels = train_test_split(
    features,
    labels,
    test_size=0.2,
    random_state=42,
)

# 3. Scale the measurements so they are on a similar range.
scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
test_features = scaler.transform(test_features)

# 4. Create the ANN (MLP means Multi-Layer Perceptron).
# This network has one hidden layer containing 10 neurons.
ann = MLPClassifier(
    hidden_layer_sizes=(10,),
    max_iter=2000,
    random_state=42,
)

# 5. Train the ANN with the training examples.
ann.fit(train_features, train_labels)

# 6. Ask the trained ANN to predict classes for the test examples.
predicted_labels = ann.predict(test_features)

# 7. Compare the predictions with the correct answers.
accuracy = accuracy_score(test_labels, predicted_labels)
print("Accuracy:", round(accuracy * 100, 2), "%")
print("\nConfusion matrix:")
print(confusion_matrix(test_labels, predicted_labels))
print("\nClassification report:")
print(classification_report(test_labels, predicted_labels))