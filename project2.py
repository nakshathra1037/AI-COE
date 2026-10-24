import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import keras
from keras import layers, models
from keras.utils import image_dataset_from_directory, load_img, img_to_array
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

print("CNN Dog-Cat Image Classification")
print("=" * 45)

# 1. Dataset path setup
# Automatically locates the local Cats vs Dogs dataset or downloads it via kagglehub
def get_dataset_paths():
    # Check local dataset folder first
    if os.path.exists("dataset/training_set") and os.path.exists("dataset/test_set"):
        return "dataset/training_set", "dataset/test_set"

    # Check local train directory if provided by user
    if os.path.exists("train"):
        return "train", None

    # Automatically fetch via kagglehub if not found locally
    print("Dataset not found locally. Fetching Cats vs Dogs dataset via kagglehub...")
    import kagglehub
    cached_path = kagglehub.dataset_download("chetankv/dogs-cats-images")
    train_dir = os.path.join(cached_path, "dataset", "training_set")
    test_dir = os.path.join(cached_path, "dataset", "test_set")
    return train_dir, test_dir

TRAIN_DIR, TEST_DIR = get_dataset_paths()
print(f"Training data path: {TRAIN_DIR}")
if TEST_DIR:
    print(f"Test data path: {TEST_DIR}")

# 2. Load images into train, validation and test datasets
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32

if TEST_DIR and os.path.exists(TEST_DIR):
    # Training set (80% of training_set)
    train_dataset = image_dataset_from_directory(
        TRAIN_DIR,
        validation_split=0.20,
        subset="training",
        seed=42,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary"
    )

    # Validation set (20% of training_set)
    val_dataset = image_dataset_from_directory(
        TRAIN_DIR,
        validation_split=0.20,
        subset="validation",
        seed=42,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary"
    )

    # Dedicated test set
    test_dataset = image_dataset_from_directory(
        TEST_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
        shuffle=False
    )
else:
    # Single directory fallback: 70% train, 15% val, 15% test
    full_train = image_dataset_from_directory(
        TRAIN_DIR,
        validation_split=0.30,
        subset="training",
        seed=42,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary"
    )
    val_test = image_dataset_from_directory(
        TRAIN_DIR,
        validation_split=0.30,
        subset="validation",
        seed=42,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
        shuffle=False
    )
    val_batches = tf.data.experimental.cardinality(val_test) // 2
    val_dataset = val_test.take(val_batches)
    test_dataset = val_test.skip(val_batches)
    train_dataset = full_train

# Optimize pipeline performance with prefetching and caching
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.cache().prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.cache().prefetch(buffer_size=AUTOTUNE)

# 3. Build CNN with exactly THREE convolutional layers
model = models.Sequential([
    layers.Input(shape=(128, 128, 3)),
    layers.Rescaling(1.0 / 255),  # Efficiently normalizes pixel values [0, 255] -> [0, 1]

    # Convolutional layer 1
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    # Convolutional layer 2
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    # Convolutional layer 3
    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(1, activation="sigmoid")
])

# 4. Compile model
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# 5. Train model
history = model.fit(
    train_dataset,
    epochs=5,
    validation_data=val_dataset
)

# 6. Evaluate on unseen test images
test_loss, test_accuracy = model.evaluate(
    test_dataset, verbose=0
)

print("\nTest Loss:", round(test_loss, 4))
print("Test Accuracy:", round(test_accuracy * 100, 2), "%")

# 7. Predict test labels
probabilities = model.predict(test_dataset, verbose=0).flatten()
predictions = (probabilities >= 0.5).astype(int)

# Extract ground truth labels from test dataset
y_test = []
for _, labels in test_dataset:
    y_test.extend(labels.numpy().flatten())
y_test = np.array(y_test, dtype=int)

print("\nClassification Report:")
print(classification_report(
    y_test, predictions,
    labels=[0, 1],
    target_names=["Cat", "Dog"],
    zero_division=0
))

# 8. Display confusion matrix
cm = confusion_matrix(y_test, predictions, labels=[0, 1])
ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Cat", "Dog"]
).plot()
plt.title("CNN Confusion Matrix")
plt.show()

# 9. Plot training and validation accuracy
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("CNN Model Accuracy")
plt.legend()
plt.show()

# 10. Save trained model
model.save("dog_cat_cnn.keras")
print("Model saved as dog_cat_cnn.keras")

# 11. Predict one image
def predict_animal(image_path):
    img = load_img(
        image_path, target_size=(128, 128)
    )
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    probability = model.predict(img_array, verbose=0)[0][0]

    if probability >= 0.5:
        print(f"Prediction: Dog ({probability * 100:.2f}%)")
    else:
        print(f"Prediction: Cat ({(1-probability) * 100:.2f}%)")

# Example: uncomment and replace with a real image path to test
# predict_animal("dataset/test_set/cats/cat.4001.jpg")
