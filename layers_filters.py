# cannot easily visualize filters lower down
from matplotlib import pyplot
import tensorflow
import numpy as np
import matplotlib.pyplot as plt


# def filters(model):
#     # retrieve weights from the second hidden layer
#     filters, biases = model.layers[20].get_weights()
#     # normalize filter values to 0-1 so we can visualize them
#     f_min, f_max = filters.min(), filters.max()
#     filters = (filters - f_min) / (f_max - f_min)
#     # plot first few filters
#     n_filters, ix = 6, 1
#     for i in range(n_filters):
#         # get the filter
#         f = filters[:, :, :, i]
#         # plot each channel separately
#         for j in range(3):
#             # specify subplot and turn of axis
#             ax = pyplot.subplot(n_filters, 3, ix)
#             ax.set_xticks([])
#             ax.set_yticks([])
#             # plot filter channel in grayscale
#             pyplot.imshow(f[:, :, j], cmap='gray')
#             ix += 1
#     # show the figure
#     pyplot.show()


# visualize feature maps output from each block in the vgg model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import Model
from matplotlib import pyplot
from numpy import expand_dims


# load the model
def feature_map(model, ixs, cmap=None):
    # redefine model to output right after the first hidden layer
    outputs = [model.layers[i].output for i in ixs]
    model = Model(inputs=model.inputs, outputs=outputs)
    # load the image with the required shape
    img = load_img('dataset_test/apple/apple13.jpg', target_size=(227, 227))
    # img = load_img('dogs_test/Japanese_spaniel/Japanese_spaniel0.jpg', target_size=(227, 227))
    # convert the image to an array
    img = img_to_array(img)
    # expand dimensions so that it represents a single 'sample'
    img = expand_dims(img, axis=0)
    # prepare the image (e.g. scale pixel values for the vgg)
    img = preprocess_input(img)
    # get feature map for first hidden layer
    feature_maps = model.predict(img)
    # plot the output from each block
    square = 8
    for fmap in feature_maps:
        # plot all 64 maps in an 8x8 squares
        ix = 1
        for _ in range(square):
            for _ in range(square):
                # specify subplot and turn of axis
                ax = pyplot.subplot(square, square, ix)
                ax.set_xticks([])
                ax.set_yticks([])
                # plot filter channel in grayscale
                pyplot.imshow(fmap[0, :, :, ix - 1], cmap='viridis')
                ix += 1
        # show the figure
        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')
        pyplot.show()


def func(model):
    # Get list of layers from model
    layer_outputs = [layer.output for layer in model.layers[1:]]
    # Create a visualization model
    visualize_model = tensorflow.keras.models.Model(inputs=model.input, outputs=layer_outputs)
    # Load image for prediction
    img = load_img('dogs_test/Japanese_spaniel/Japanese_spaniel0.jpg', target_size=(227, 227))
    # Convert image to array
    x = img_to_array(img)
    x = x.reshape((1, 227, 227, 3))

    # Rescale the image
    x = x / 255

    # Get all layers feature maps for image
    feature_maps = visualize_model.predict(x)

    # Show names of layers available in model
    layer_names = [layer.name for layer in model.layers]
    print(layer_names)

    # Plotting the graph
    for layer_names, feature_maps in zip(layer_names, feature_maps):
        print(feature_maps.shape)
        if len(feature_maps.shape) == 4:
            channels = feature_maps.shape[-1]
            size = feature_maps.shape[1]
            display_grid = np.zeros((size, size * channels))
            for i in range(channels):
                x = feature_maps[0, :, :, i]
                x -= x.mean()
                x /= x.std()
                x *= 64
                x += 128
                x = np.clip(x, 0, 255).astype('uint8')
                # We'll tile each filter into this big horizontal grid
                display_grid[:, i * size: (i + 1) * size] = x

            scale = 20. / channels
            plt.figure(figsize=(scale * channels, scale))
            plt.title(layer_names)
            plt.grid(False)
            plt.imshow(display_grid, aspect='auto', cmap='viridis')
