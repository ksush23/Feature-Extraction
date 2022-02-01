import tensorflow as tf
import datetime
import pathlib
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications import MobileNet
import numpy as np


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get("accuracy") == 1.00 and logs.get("loss") < 0.03:
            print("\nReached 100% accuracy so stopping training")
            self.model.stop_training = True


def build_model(data_dir, CLASS_NAMES, model_name):
    image_count = len(list(data_dir.glob('*/*.jpg')))
    output_class_units = len(CLASS_NAMES)

    # Shape of inputs to NN Model
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
                                                         # Resizing the raw dataset
                                                         classes=list(CLASS_NAMES))

    base_model = MobileNet(weights='imagenet',
                           include_top=False)  # imports the mobilenet model and discards the last 1000 neuron layer.

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(
        x)  # we add dense layers so that the model can learn more complex functions and classify for better results.
    x = Dense(1024, activation='relu')(x)  # dense layer 2
    x = Dense(512, activation='relu')(x)  # dense layer 3
    preds = Dense(output_class_units, activation='softmax')(x)  # final layer with softmax activation
    model = Model(inputs=base_model.input, outputs=preds)
    for layer in model.layers[:20]:
        layer.trainable = False
    for layer in model.layers[20:]:
        layer.trainable = True

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

    callbacks = myCallback()
    log_dir = "logs\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    model.fit_generator(generator=train_data_gen,
                        steps_per_epoch=STEPS_PER_EPOCH,
                        epochs=20,
                        callbacks=[tensorboard_callback, callbacks])

    # Saving the model
    model.save(model_name)


if __name__ == '__main__':
    separable = False
    if separable:
        dir = pathlib.Path("./dataset_train")
        name = "MobileNet.h5"
    else:
        dir = pathlib.Path("./dogs_train")
        name = "MobileNet_non_separable.h5"
    classes = np.array([item.name for item in dir.glob('*')])
    build_model(dir, classes, name)
