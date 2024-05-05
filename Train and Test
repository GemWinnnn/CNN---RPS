import tensorflow as tf
import tensorflow_datasets as tfds
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import load_model

# Hyperparameters
batch_size = 32
num_classes = 3
epochs = 5

# Ensure the 'results' folder exists
if not os.path.isdir("results"):
    os.mkdir("results")

# Load and preprocess the rock_paper_scissors dataset
def load_data():
    def preprocess_image(image, label):
        image = tf.image.convert_image_dtype(image, tf.float32)  # Normalize
        return image, label

    ds_train, info = tfds.load("rock_paper_scissors", with_info=True, split="train", as_supervised=True)
    ds_test = tfds.load("rock_paper_scissors", split="test", as_supervised=True)

    ds_train = ds_train.shuffle(1024).map(preprocess_image).batch(batch_size)
    ds_test = ds_test.shuffle(1024).map(preprocess_image).batch(batch_size)
    return ds_train, ds_test, info

# Create the model architecture
def create_model():
    model = Sequential([
        AveragePooling2D(pool_size=6, strides=3, input_shape=(300, 300, 3)),
        Conv2D(64, 3, activation='relu'),
        Conv2D(32, 3, activation='relu'),
        MaxPooling2D(2, 2),
        Dropout(0.5),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        loss="sparse_categorical_crossentropy", 
        optimizer="adam", 
        metrics=["accuracy"]
    )
    return model

# Train the model and capture history
def train_model(model, ds_train, ds_test, info):
    tensorboard = TensorBoard(log_dir="logs/rps-model")

    history = model.fit(
        ds_train,
        epochs=epochs,
        validation_data=ds_test,
        verbose=1,
        steps_per_epoch=info.splits["train"].num_examples // batch_size,
        validation_steps=info.splits["test"].num_examples // batch_size,
        callbacks=[tensorboard]
    )

    return history

# Function to plot loss and accuracy over epochs
def plot_training_history(history):
    plt.figure(figsize=(12, 6))
    
    # Plotting loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plotting accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()  # Adjusts subplot params for better appearance
    plt.savefig("results/accuracy_loss_plot.png")  # Save plot to the 'results' directory
    plt.show()  # Display the plot

# Load data
ds_train, ds_test, info = load_data()

# Create and train the model
model = create_model()
history = train_model(model, ds_train, ds_test, info)

# Plot accuracy and loss over epochs
plot_training_history(history)

# Save the trained model
model.save("RPSmodel.h5")

# Load the saved model for predictions
loaded_model = load_model("RPSmodel.h5")

# Get a batch of test data and predict on the first 5 images
test_batch = next(iter(ds_test))
sample_images = test_batch[0][:5]  # Get the first 5 images
sample_labels = test_batch[1][:5].numpy()  # Corresponding labels

# Predict labels for the images
predictions = loaded_model.predict(sample_images)

# Display and save predictions along with true labels
plt.figure(figsize=(12, 4))  # Adjust the figure size for better layout
for i in range(5):
    plt.subplot(1, 5, i + 1)  # Create subplots for 5 images in one row
    plt.axis('off')  # Hide the axes
    plt.imshow(sample_images[i].numpy())  # Show the image
    true_label = {0: "rock", 1: "paper", 2: "scissors"}[sample_labels[i]]
    predicted_label = {0: "rock", 1: "paper", 2: "scissors"}[np.argmax(predictions[i])]
    plt.title(f"True: {true_label}\nPred: {predicted_label}")  # Titles with true and predicted labels

# Save and display the prediction plot
plt.savefig("results/prediction_plot.png")
plt.show()
