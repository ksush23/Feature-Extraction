import pathlib
import numpy as np
from test import test_accuracy
from rgb_extractor import get_accuracy
from visualization import visualize

# Separable classes
separable = True
if separable:
    # Load dataset directory
    data_dir_train = pathlib.Path("./dataset_train")
    image_count_train = len(list(data_dir_train.glob('*/*.jpg')))
    print("Number of instances in train dataset: ", image_count_train)
    # classnames in the dataset specified
    CLASS_NAMES = np.array([item.name for item in data_dir_train.glob('*')])
    print(CLASS_NAMES)

    data_dir_test = pathlib.Path("./dataset_test")
    image_count_test = len(list(data_dir_test.glob('*/*.jpg')))
    print("Number of instances in test dataset: ", image_count_test)

    df_pca_non_separable = "df_pca.pkl"
    df_tsne_non_separable = "df_tsne.pkl"
    visualize(df_pca_non_separable, df_tsne_non_separable, len(CLASS_NAMES))

    target_names_separable = ['Apple', 'Banana', 'Orange']
    print("Own (rgb) feature extractor: ")
    filename_separable = 'rgb_extractor_model_.sav'
    get_accuracy(3, filename_separable, target_names_separable)

    print("AlexNet: ")
    alexnet_model_name = "AlexNet.h5"
    test_accuracy(CLASS_NAMES, data_dir_test, alexnet_model_name, target_names_separable)

    print("MobileNet: ")
    mobilenet_model_name = "MobileNet.h5"
    test_accuracy(CLASS_NAMES, data_dir_test, mobilenet_model_name, target_names_separable)

    print("Xception: ")
    xception_model_name = "Xception.h5"
    test_accuracy(CLASS_NAMES, data_dir_test, xception_model_name, target_names_separable)

else:
    # Hardly separable classes

    # # Load dataset directory
    data_dir_train = pathlib.Path("./dogs_train")
    image_count_train = len(list(data_dir_train.glob('*/*.jpg')))
    print("Number of instances in train dataset: ", image_count_train)
    # classnames in the dataset specified
    CLASS_NAMES = np.array([item.name for item in data_dir_train.glob('*')])
    print(CLASS_NAMES)

    data_dir_test = pathlib.Path("./dogs_test")
    image_count_test = len(list(data_dir_test.glob('*/*.jpg')))
    print("Number of instances in test dataset: ", image_count_test)

    df_pca_non_separable = "df_pca_non-separable.pkl"
    df_tsne_non_separable = "df_tsne-non-separable.pkl"
    visualize(df_pca_non_separable, df_tsne_non_separable, len(CLASS_NAMES))
    target_names_non_separable = ['Blenheim_spaniel', 'Japanese_spaniel']
    print("Own (rgb) feature extractor: ")
    filename_non_separable = 'rgb_extractor_model_non-separable.sav'
    get_accuracy(2, filename_non_separable, target_names_non_separable)

    print("AlexNet: ")
    alexnet_model_name = "AlexNet_non_separable.h5"
    test_accuracy(CLASS_NAMES, data_dir_test, alexnet_model_name, target_names_non_separable)

    print("MobileNet: ")
    mobilenet_model_name = "MobileNet_non_separable.h5"
    test_accuracy(CLASS_NAMES, data_dir_test, mobilenet_model_name, target_names_non_separable)

    print("Xception: ")
    xception_model_name = "Xception_non_separable.h5"
    test_accuracy(CLASS_NAMES, data_dir_test, xception_model_name, target_names_non_separable)
