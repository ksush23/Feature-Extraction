import tensorflow as tf
import numpy as np
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from layers_filters import feature_map, func


def test_accuracy(CLASS_NAMES, data_dir, model_name, target_names):
    # preprocess the data
    IMG_HEIGHT = 227  # input Shape required by the model
    IMG_WIDTH = 227  # input Shape required by the model

    image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
    test_data_gen = image_generator.flow_from_directory(directory=str(data_dir),
                                                        shuffle=True,
                                                        batch_size=1,
                                                        target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                        classes=list(CLASS_NAMES))

    # Loading the saved model
    new_model = tf.keras.models.load_model(model_name)
    new_model.summary()
    loss, acc = new_model.evaluate(test_data_gen)

    print("accuracy:{:.2f}%".format(acc * 100))

    image_count_test = len(list(data_dir.glob('*/*.jpg')))
    print_confusion_matrix(new_model, test_data_gen, image_count_test, target_names, model_name)
    # filters(new_model)

    # if new_model.name == "AlexNet":
    #     ixs = [2, 5, 9, 15, 18]
    # if "Xception" in model_name:
    #     ixs = [20, 27, 33, 40, 46, 96]
    # if "MobileNet" in model_name:
    #     ixs = [20, 27, 33, 40, 48, 85]
    # feature_map(new_model, ixs)


def print_confusion_matrix(model, test_generator, size, target_names, model_name):
    # Confution Matrix and Classification Report
    Y_pred = model.predict(test_generator, size)  # // batch_size + 1)
    y_pred = np.argmax(Y_pred, axis=1)
    print('Confusion Matrix')
    df = confusion_matrix(test_generator.classes, y_pred)
    print(df)
    print('Classification Report')
    print(classification_report(test_generator.classes, y_pred, target_names=target_names))
