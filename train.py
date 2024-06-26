from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.callbacks import TensorBoard
import tensorflow as tf
import tensorflow_datasets as tfds
import os
import matplotlib.pyplot as plt  # Added for plotting

# Setting hyper-parameters
batch_size = 32
num_classes = 3
epochs = 5

"""Loading and preparing the data"""

# Loading rock_paper_scissors dataset and preprocessing it
def load_data():
    def preprocess_image(image, label):
        # convert [0, 255] range integers to [0, 1] range floats
        image = tf.image.convert_image_dtype(image, tf.float32)
        return image, label

    # loading the dataset, split into train and test
    ds_train, info = tfds.load("rock_paper_scissors", with_info=True, split="train", as_supervised=True)
    ds_test = tfds.load("rock_paper_scissors", split="test", as_supervised=True)

    # repeat dataset forever, shuffle the images, preprocess the images, split by batch
    ds_train = ds_train.repeat().shuffle(1024).map(preprocess_image).batch(batch_size)
    ds_test = ds_test.repeat().shuffle(1024).map(preprocess_image).batch(batch_size)
    return ds_train, ds_test, info


"""Creating the model"""

def create_model():
    model = Sequential()
    model.add(AveragePooling2D(6, 3, input_shape=(300, 300, 3)))
    model.add(Conv2D(64, 3, activation='relu'))
    model.add(Conv2D(32, 3, activation='relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.summary()  # print the summary of the model architecture
    # training the model using adam optimizer
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


"""Execute our stuff: load data, initiate model, train model"""

# load data
ds_train, ds_test, info = load_data()

# create the model
model = create_model()

# callbacks for fitting our model
logdir = os.path.join("logs", "rps-model")
tensorboard = TensorBoard(log_dir=logdir)

# create 'results' folder if it doesn't exist yet
if not os.path.isdir("results"):
    os.mkdir("results")

# train the model and capture history
history = model.fit(
    ds_train,
    epochs=epochs,
    validation_data=ds_test,
    verbose=1,
    steps_per_epoch=info.splits["train"].num_examples // batch_size,
    validation_steps=info.splits["test"].num_examples // batch_size,
    callbacks=[tensorboard]
)

"""Plotting and saving the model loss"""
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.savefig("results/model_loss_plot.png")  # Save the plot as an image
plt.show()

"""Plotting and saving the model accuracy"""
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='lower right')
plt.savefig("results/model_accuracy_plot.png")  # Save the plot as an image
plt.show()

"""Evaluate and save the model"""
# load the test data
rock_paper_scissors_test = tfds.load(name="rock_paper_scissors", split='test', batch_size=-1)

# make NumPy array records out of a tf.data.Dataset
rock_paper_scissors_test

# separate the x (input image) and y (output label)
x_test, y_test = rock_paper_scissors_test["image"], rock_paper_scissors_test["label"]

# evaluate the model
model.evaluate(x_test, y_test)

# save the model
model.save("RPSmodel.h5")
