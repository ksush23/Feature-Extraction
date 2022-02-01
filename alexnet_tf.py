# import necessary package
import tensorflow as tf
import numpy as np
import pathlib
import datetime
from keras.layers import Input, Dense, Activation,BatchNormalization, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.models import Model


# callbacks at training
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get("accuracy") == 1.00 and logs.get("loss") < 0.03:
            print("\nReached 100% accuracy so stopping training")
            self.model.stop_training = True


def build_model(data_dir, CLASS_NAMES, model_name):
    image_count = len(list(data_dir.glob('*/*.jpg')))
    output_class_units = len(CLASS_NAMES)

    BATCH_SIZE = 32  # Can be of size 2^n, but not restricted to. for the better utilization of memory
    IMG_HEIGHT = 227  # input Shape required by the model
    IMG_WIDTH = 227  # input Shape required by the model
    STEPS_PER_EPOCH = np.ceil(image_count / BATCH_SIZE)

    # Rescalingthe pixel values from 0~255 to 0~1 For RGB Channels of the image.
    image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
    # training_data for model training
    train_data_gen = image_generator.flow_from_directory(directory=str(data_dir),
                                                         batch_size=BATCH_SIZE,
                                                         shuffle=True,
                                                         target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                         classes=list(CLASS_NAMES))

    X_input = Input((227, 227, 3))

    X = Conv2D(96, (11, 11), strides=(4, 4), name="conv0")(X_input)
    X = BatchNormalization(axis=3, name="bn0")(X)
    X = Activation('relu')(X)

    X = MaxPooling2D((3, 3), strides=(2, 2), name='max0')(X)

    X = Conv2D(256, (5, 5), strides=(1, 1), padding='same', name='conv1')(X)
    X = BatchNormalization(axis=3, name='bn1')(X)
    X = Activation('relu')(X)

    X = MaxPooling2D((3, 3), strides=(2, 2), name='max1')(X)

    X = Conv2D(384, (3, 3), strides=(1, 1), padding='same', name='conv2')(X)
    X = BatchNormalization(axis=3, name='bn2')(X)
    X = Activation('relu')(X)

    X = Conv2D(384, (3, 3), strides=(1, 1), padding='same', name='conv3')(X)
    X = BatchNormalization(axis=3, name='bn3')(X)
    X = Activation('relu')(X)

    X = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='conv4')(X)
    X = BatchNormalization(axis=3, name='bn4')(X)
    X = Activation('relu')(X)

    X = MaxPooling2D((3, 3), strides=(2, 2), name='max2')(X)

    X = Flatten()(X)

    X = Dense(4096, activation='relu', name="fc0")(X)
    X = Dropout(0.5)(X)

    X = Dense(4096, activation='relu', name='fc1')(X)
    X = Dropout(0.5)(X)

    X = Dense(output_class_units, activation='softmax', name='fc2')(X)

    model = Model(inputs=X_input, outputs=X, name='AlexNet')

    # Specifying the optimizer, Loss function for optimization & Metrics to be displayed
    model.compile(optimizer='sgd', loss="categorical_crossentropy", metrics=['accuracy'])

    # Summarizing the model architecture and printing it out
    model.summary()

    callbacks = myCallback()

    # TensorBoard.dev Visuals
    log_dir = "logs\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Training the Model
    history = model.fit(
        train_data_gen,
        steps_per_epoch=STEPS_PER_EPOCH,
        epochs=30,
        callbacks=[tensorboard_callback, callbacks])
    # Saving the model
    model.save(model_name)


if __name__ == '__main__':
    separable = True
    if separable:
        dir = pathlib.Path("./dataset_train")
        name = "AlexNet.h5"
    else:
        dir = pathlib.Path("./dogs_train")
        name = "AlexNet_non_separable.h5"
    classes = np.array([item.name for item in dir.glob('*')])
    build_model(dir, classes, name)
