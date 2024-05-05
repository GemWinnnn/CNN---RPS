import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import os

# Setting hyper-parameters
batch_size = 32
num_classes = 3
epochs = 3

# Ensure the 'results' folder exists
if not os.path.isdir("results"):
    os.mkdir("results")

# Load the data
def load_data():
    def preprocess_image(image, label):
        # convert [0, 255] range integers to [0, 1] range floats
        image = tf.image.convert_image_dtype(image, tf.float32)
        return image, label

    # load the dataset, split into train and test
    ds_train, info = tfds.load("rock_paper_scissors", with_info=True, split="train", as_supervised=True)
    ds_test = tfds.load("rock_paper_scissors", split="test", as_supervised=True)

    # shuffle and preprocess the test set
    ds_test = ds_test.shuffle(1024).map(preprocess_image).batch(batch_size)
    return ds_test, info

# Load test data and the model
ds_test, info = load_data()
categories = {0: "rock", 1: "paper", 2: "scissors"}
loaded_model = load_model("RPSmodel.h5")

# Get a batch of test data and predict on the first 5 images
test_batch = next(iter(ds_test))
sample_images = test_batch[0][:5]  # Get the first 5 images
sample_labels = test_batch[1][:5].numpy()  # Corresponding labels

# Predict labels for the images
predictions = loaded_model.predict(sample_images)

# Display and save the results
plt.figure(figsize=(12, 4))  # Adjust the figure size for better layout
for i in range(5):
    plt.subplot(1, 5, i + 1)  # Create subplots for 5 images in one row
    plt.axis('off')  # Hide the axes
    plt.imshow(sample_images[i].numpy())  # Show the image
    true_label = categories[sample_labels[i]]
    predicted_label = categories[np.argmax(predictions[i])]
    plt.title(f"True: {true_label}\nPred: {predicted_label}")  # Titles with true and predicted labels

# Save the plot to 'results' folder
plt.savefig("results/prediction_plot.png")  # Save the plot as an image
plt.show()  # Display the plots
