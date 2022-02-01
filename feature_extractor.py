import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from pylab import *
import numpy as np
import pathlib
import matplotlib.image as mplib

data_dir_apple = pathlib.Path("./dataset_train/apple")
list_apple = np.array([item for item in data_dir_apple.glob('*')])

data_dir_banana = pathlib.Path("./dataset_train/banana")
list_banana = np.array([item for item in data_dir_banana.glob('*')])

data_dir_orange = pathlib.Path("./dataset_train/orange")
list_orange = np.array([item for item in data_dir_orange.glob('*')])

pca = PCA(n_components=10)
tsne = TSNE(n_components=10, verbose=1, perplexity=40, n_iter=300)

pca_apple = np.empty(0, float)
tsne_apple = np.empty(0, float)
for item in list_apple:
    img = mplib.imread(item)
    img_r = np.reshape(img, (img.shape[0], img.shape[1] * img.shape[2]))
    feat_cols = ['pixel'+str(i) for i in range(img_r.shape[1])]
    df = pd.DataFrame(img_r, columns=feat_cols)
    pca_result = np.append(np.array(pca.fit_transform(df[feat_cols].values)).flatten(), np.array([1]), axis=0)
    pca_apple = np.append(pca_apple, pca_result, axis=0)
    tsne_results = np.array(tsne.fit_transform(df[feat_cols].values)).flatten()
    tsne_apple = np.append(np.append(tsne_apple, tsne_results, axis=0), 1, axis=0)

pca_banana = np.empty(0, float)
tsne_banana = np.empty(0, float)
for item in list_banana:
    img = mplib.imread(item)
    img_r = np.reshape(img, (img.shape[0], img.shape[1] * img.shape[2]))
    feat_cols = ['pixel'+str(i) for i in range(img_r.shape[1])]
    df = pd.DataFrame(img_r, columns=feat_cols)
    pca_result = np.array(pca.fit_transform(df[feat_cols].values)).flatten()
    tsne_results = np.array(tsne.fit_transform(df[feat_cols].values)).flatten()
    pca_banana = np.append(pca_banana, pca_result, axis=0)
    tsne_banana = np.append(tsne_banana, tsne_results, axis=0)

pca_orange = np.empty(0, float)
tsne_orange = np.empty(0, float)
for item in list_orange:
    img = mplib.imread(item)
    img_r = np.reshape(img, (img.shape[0], img.shape[1] * img.shape[2]))
    feat_cols = ['pixel'+str(i) for i in range(img_r.shape[1])]
    df = pd.DataFrame(img_r, columns=feat_cols)
    pca_result = np.array(pca.fit_transform(df[feat_cols].values)).flatten()
    tsne_results = np.array(tsne.fit_transform(df[feat_cols].values)).flatten()
    pca_orange = np.append(pca_orange, pca_result, axis=0)
    tsne_orange = np.append(tsne_orange, tsne_results, axis=0)
