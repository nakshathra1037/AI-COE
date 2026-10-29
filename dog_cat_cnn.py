
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Dataset path
train_path = "dogs_vs_cats_subset/train"
test_path = "dogs_vs_cats_subset/val"

# Load training dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    "dogs_vs_cats_subset/train",
    image_size=(128, 128),
    batch_size=32
)

# Load validation dataset
val_ds = tf.keras.utils.image_dataset_from_directory(
    test_path,
    image_size=(128, 128),
    batch_size=32
)

# Load test dataset
test_ds = tf.keras.utils.image_dataset_from_directory(
    "dogs_vs_cats_subset/val",
    image_size=(128, 128),
    batch_size=32
)

# Create CNN model with three convolutional layers
model = models.Sequential([
    layers.Input(shape=(128, 128, 3)),
    layers.Rescaling(1.0 / 255),

    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

# Compile model
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Display model architecture
model.summary()

# Train model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5
)

# Test model
loss, accuracy = model.evaluate(test_ds)
print("Test Accuracy:", round(accuracy * 100, 2), "%")

# Plot accuracy graph
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Dog-Cat CNN Accuracy")
plt.legend()
plt.show()

# Save model
model.save("dog_cat_model.keras")
print("Model saved successfully!")
