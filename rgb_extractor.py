from PIL import Image
import pathlib
import numpy as np
from sklearn import svm
from sklearn import metrics
import pickle
from sklearn.metrics import classification_report, confusion_matrix


def get_rgb_vector(photo_link):
    red_0 = 0
    red_1 = 0
    red_2 = 0
    red_3 = 0

    green_0 = 0
    green_1 = 0
    green_2 = 0
    green_3 = 0

    blue_0 = 0
    blue_1 = 0
    blue_2 = 0
    blue_3 = 0

    im = Image.open(photo_link)
    pix = im.load()
    for i in range(im.size[0]):
        for j in range(im.size[1]):
            red, green, blue = pix[i, j]
            red_0, red_1, red_2, red_3 = get_range(red, red_0, red_1, red_2, red_3)
            green_0, green_1, green_2, green_3 = get_range(green, green_0, green_1, green_2, green_3)
            blue_0, blue_1, blue_2, blue_3 = get_range(blue, blue_0, blue_1, blue_2, blue_3)
    return np.array([red_0, red_1, red_2, red_3, green_0, green_1, green_2, green_3, blue_0, blue_1, blue_2, blue_3])


def get_range(color, color_0, color_1, color_2, color_3):
    if color < 64:
        return 1 + color_0, color_1, color_2, color_3
    elif color < 128:
        return color_0, 1 + color_1, color_2, color_3
    elif color < 192:
        return color_0, color_1, 1 + color_2, color_3
    else:
        return color_0, color_1, color_2, 1 + color_3


def get_accuracy(classes, filename, target_names):
    if classes == 3:
        data_dir_blenheim_test = pathlib.Path("./dataset_test/apple")
        list_blenheim_test = np.array([item for item in data_dir_blenheim_test.glob('*')])

        data_dir_japanese_test = pathlib.Path("./dataset_test/banana")
        list_japanese_test = np.array([item for item in data_dir_japanese_test.glob('*')])

        data_dir_orange_test = pathlib.Path("./dataset_test/orange")
        list_orange_test = np.array([item for item in data_dir_orange_test.glob('*')])

        test_blenheim_x = np.empty((0, 12), int)
        for item in list_blenheim_test:
            test_blenheim_x = np.append(test_blenheim_x, np.array([get_rgb_vector(item)]), axis=0)
        test_blenheim_y = [1 for i in range(len(test_blenheim_x))]

        test_japanese_x = np.empty((0, 12), int)
        for item in list_japanese_test:
            test_japanese_x = np.append(test_japanese_x, np.array([get_rgb_vector(item)]), axis=0)
        test_japanese_y = [0 for i in range(len(test_japanese_x))]

        test_orange_x = np.empty((0, 12), int)
        for item in list_orange_test:
            test_orange_x = np.append(test_orange_x, np.array([get_rgb_vector(item)]), axis=0)
        test_orange_y = [-1 for i in range(len(test_orange_x))]

        x_test = np.append(np.append(test_blenheim_x, test_japanese_x, axis=0), test_orange_x, axis=0)
        y_test = np.append(np.append(test_blenheim_y, test_japanese_y, axis=0), test_orange_y, axis=0)

    else:
        data_dir_blenheim_test = pathlib.Path("./dogs_test/Blenheim_spaniel")
        list_blenheim_test = np.array([item for item in data_dir_blenheim_test.glob('*')])

        data_dir_japanese_test = pathlib.Path("./dogs_test/Japanese_spaniel")
        list_japanese_test = np.array([item for item in data_dir_japanese_test.glob('*')])

        test_blenheim_x = np.empty((0, 12), int)
        for item in list_blenheim_test:
            test_blenheim_x = np.append(test_blenheim_x, np.array([get_rgb_vector(item)]), axis=0)
        test_blenheim_y = [1 for i in range(len(test_blenheim_x))]

        test_japanese_x = np.empty((0, 12), int)
        for item in list_japanese_test:
            test_japanese_x = np.append(test_japanese_x, np.array([get_rgb_vector(item)]), axis=0)
        test_japanese_y = [0 for i in range(len(test_japanese_x))]

        x_test = np.append(test_blenheim_x, test_japanese_x, axis=0)
        y_test = np.append(test_blenheim_y, test_japanese_y, axis=0)

    loaded_model = pickle.load(open(filename, 'rb'))
    # Predict the response for test dataset
    y_pred = loaded_model.predict(x_test)

    # Model Accuracy: how often is the classifier correct?
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred) * 100)

    print('Confusion Matrix')
    df = confusion_matrix(y_test, y_pred)
    print(df)
    print('Classification Report')
    print(classification_report(y_test, y_pred, target_names=target_names))


if __name__ == '__main__':
    separable = False
    if separable:
        data_dir_blenheim_train = pathlib.Path("./dataset_train/apple")
        list_blenheim_train = np.array([item for item in data_dir_blenheim_train.glob('*')])

        data_dir_japanese_train = pathlib.Path("./dataset_train/banana")
        list_japanese_train = np.array([item for item in data_dir_japanese_train.glob('*')])

        data_dir_orange_train = pathlib.Path("./dataset_train/orange")
        list_orange_train = np.array([item for item in data_dir_orange_train.glob('*')])

        print("Start getting training data for apple")
        train_blenheim_x = np.empty((0, 12), int)
        for item in list_blenheim_train:
            train_blenheim_x = np.append(train_blenheim_x, np.array([get_rgb_vector(item)]), axis=0)
        train_blenheim_y = [1 for i in range(len(train_blenheim_x))]

        print("Start getting training data for banana")
        train_japanese_x = np.empty((0, 12), int)
        for item in list_japanese_train:
            train_japanese_x = np.append(train_japanese_x, np.array([get_rgb_vector(item)]), axis=0)
        train_japanese_y = [0 for i in range(len(train_japanese_x))]

        print("Start getting training data for orange")
        train_orange_x = np.empty((0, 12), int)
        for item in list_orange_train:
            train_orange_x = np.append(train_orange_x, np.array([get_rgb_vector(item)]), axis=0)
        train_orange_y = [-1 for i in range(len(train_orange_x))]

        x_train = np.append(np.append(train_blenheim_x, train_japanese_x, axis=0), train_orange_x, axis=0)
        y_train = np.append(np.append(train_blenheim_y, train_japanese_y, axis=0), train_orange_y, axis=0)
        filename = 'rgb_extractor_model_.sav'

    else:
        data_dir_blenheim_train = pathlib.Path("./dogs_train/Blenheim_spaniel")
        list_blenheim_train = np.array([item for item in data_dir_blenheim_train.glob('*')])

        data_dir_japanese_train = pathlib.Path("./dogs_train/Japanese_spaniel")
        list_japanese_train = np.array([item for item in data_dir_japanese_train.glob('*')])

        print("Start getting training data for blenheim spaniel")
        train_blenheim_x = np.empty((0, 12), int)
        for item in list_blenheim_train:
            train_blenheim_x = np.append(train_blenheim_x, np.array([get_rgb_vector(item)]), axis=0)
        train_blenheim_y = [1 for i in range(len(train_blenheim_x))]

        print("Start getting training data for japanese spaniel")
        train_japanese_x = np.empty((0, 12), int)
        for item in list_japanese_train:
            train_japanese_x = np.append(train_japanese_x, np.array([get_rgb_vector(item)]), axis=0)
        train_japanese_y = [0 for i in range(len(train_japanese_x))]

        x_train = np.append(train_blenheim_x, train_japanese_x, axis=0)
        y_train = np.append(train_blenheim_y, train_japanese_y, axis=0)
        filename = 'rgb_extractor_model_non-separable.sav'

    print("Creating svm classifier")
    # Create a svm Classifier
    clf = svm.SVC(kernel='linear')

    # Train the model using the training sets
    clf.fit(x_train, y_train)
    pickle.dump(clf, open(filename, 'wb'))
