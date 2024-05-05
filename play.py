import cv2
from keras.models import load_model
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Rock-paper-scissors classes
categories = {
    0: "rock",
    1: "paper",
    2: "scissors"
}

# Load the model
model = load_model("RPSmodel.h5")

# Function to load and preprocess the test dataset
def load_data():
    def preprocess_image(image, label):
        # Convert [0, 255] range integers to [0, 1] range floats
        image = tf.image.convert_image_dtype(image, tf.float32)
        return image, label

    ds_train, info = tfds.load("rock_paper_scissors", with_info=True, split="train", as_supervised=True)
    ds_test = tfds.load("rock_paper_scissors", split="test", as_supervised=True)
    ds_test = ds_test.repeat().shuffle(1024).map(preprocess_image).batch(32)
    return ds_test

# Load test dataset
ds_test = load_data()

# Capture the video
vid = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = vid.read()

    # Draw a rectangle on the frame to guide where the user should place their hand
    cv2.rectangle(frame, (100, 100), (400, 400), (255, 255, 255), 2)

    # Extract the image from the rectangle
    roi = frame[100:400, 100:400]
    img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (300, 300))

    # Predict the class of the image
    prediction = np.argmax(model.predict(img.reshape(-1, *img.shape))[0])
    user_move_name = categories[prediction]

    # Display the prediction on the frame
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (110, 90)  # Position of the text
    fontScale = 1
    fontColor = (0, 255, 0)  # Green color
    lineType = 2

    cv2.putText(
        frame,
        "Predicted: " + user_move_name,
        bottomLeftCornerOfText,
        font,
        fontScale,
        fontColor,
        lineType,
    )

    # Show the frame with the prediction
    cv2.imshow('Rock-Paper-Scissors Prediction', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and destroy all windows when the loop ends
vid.release()
cv2.destroyAllWindows()
