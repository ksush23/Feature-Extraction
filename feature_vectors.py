import tensorflow as tf
import numpy as np
import pathlib
import datetime
from keras.layers import *
from keras.applications import *
from keras.models import Model


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
    LEARNING_RATE = 1e-4
    base_model = Xception(input_shape=(227, 227, 3), weights='imagenet', include_top=False)

    callbacks = myCallback()
    log_dir = "logs\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Top Model Block
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(output_class_units, activation='softmax')(x)

    # add your top layer block to your base model
    model = Model(base_model.input, predictions)
    print(model.summary())

    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer='nadam',
                  loss='categorical_crossentropy',  # categorical_crossentropy if multi-class classifier
                  metrics=['accuracy'])

    model.fit_generator(train_data_gen,
                        steps_per_epoch=STEPS_PER_EPOCH,
                        epochs=10,
                        callbacks=[tensorboard_callback, callbacks],
                        workers=0,  # tf-generators are not thread-safe
                        use_multiprocessing=False,
                        max_queue_size=0
                        )

    # Saving the model
    # model.save(model_name)


if __name__ == '__main__':
    separable = True
    if separable:
        dir = pathlib.Path("./dataset_train")
        name = "Xception_f.h5"
    else:
        dir = pathlib.Path("./dogs_train")
        name = "Xception_non_separable.h5"
    classes = np.array([item.name for item in dir.glob('*')])
    build_model(dir, classes, name)
