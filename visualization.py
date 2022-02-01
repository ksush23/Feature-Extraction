import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
from pylab import *
import numpy as np
import pathlib
import matplotlib.image as mplib


def calculate_pca_tsne(list):
    pca_list = np.empty((0, 3), float)
    tsne_list = np.empty((0, 2), float)
    for item in list:
        img = mplib.imread(item)
        img_r = np.reshape(img, (img.shape[0], img.shape[1] * img.shape[2]))
        feat_cols = ['pixel' + str(i) for i in range(img_r.shape[1])]
        df = pd.DataFrame(img_r, columns=feat_cols)
        pca = PCA(n_components=3)
        pca_result = np.array(pca.fit_transform(df[feat_cols].values))
        pca_list = np.append(pca_list, pca_result, axis=0)
        tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
        tsne_results = np.array(tsne.fit_transform(df[feat_cols].values))
        tsne_list = np.append(tsne_list, tsne_results, axis=0)
        # print(pca_result)
        # print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
    return pca_list, tsne_list


def visualize(df_pca_name, df_tsne_name, classes):
    df_pca = pd.read_pickle(df_pca_name)
    df_tsne = pd.read_pickle(df_tsne_name)
    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x="pca-one", y="pca-two",
        palette=sns.color_palette("hls", classes),
        hue="y",
        data=df_pca,
        legend="full",
        alpha=0.3
    )
    ax = plt.figure(figsize=(16, 10)).gca(projection='3d')
    ax.scatter(
        xs=df_pca["pca-one"],
        ys=df_pca["pca-two"],
        zs=df_pca["pca-three"],
        c=df_pca["y"],
        cmap='tab10'
    )
    ax.set_xlabel('pca-one')
    ax.set_ylabel('pca-two')
    ax.set_zlabel('pca-three')
    plt.show()

    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="y",
        palette=sns.color_palette("hls", classes),
        data=df_tsne,
        legend="full",
        alpha=0.3)
    plt.show()


if __name__ == "__main__":
    separable = False
    if separable:
        data_dir_blenheim = pathlib.Path("./dataset_train/apple")
        list_blenheim = np.array([item for item in data_dir_blenheim.glob('*')])

        data_dir_japanese = pathlib.Path("./dataset_train/banana")
        list_japanese = np.array([item for item in data_dir_japanese.glob('*')])

        data_dir_orange = pathlib.Path("./dataset_train/orange")
        list_orange = np.array([item for item in data_dir_orange.glob('*')])

        pca_blenheim, tsne_blenheim = calculate_pca_tsne(list_blenheim)
        pca_japanese, tsne_japanese = calculate_pca_tsne(list_japanese)
        pca_orange, tsne_orange = calculate_pca_tsne(list_orange)

        df_pca = pd.DataFrame()
        df_pca['pca-one'] = np.append(np.append(pca_blenheim[:, 0], pca_japanese[:, 0]), pca_orange[:, 0])
        df_pca['pca-two'] = np.append(np.append(pca_blenheim[:, 1], pca_japanese[:, 1]), pca_orange[:, 1])
        df_pca['pca-three'] = np.append(np.append(pca_blenheim[:, 2], pca_japanese[:, 2]), pca_orange[:, 2])
        arr_blenheim = [1 for i in range(len(pca_blenheim))]
        arr_japanese = [2 for i in range(len(pca_japanese))]
        arr_orange = [3 for i in range(len(pca_orange))]
        df_pca['y'] = np.append(np.append(arr_blenheim, arr_japanese), arr_orange)

        df_tsne = pd.DataFrame()
        df_tsne['tsne-2d-one'] = np.append(np.append(tsne_blenheim[:, 0], tsne_japanese[:, 0]), tsne_orange[:, 0])
        df_tsne['tsne-2d-two'] = np.append(np.append(tsne_blenheim[:, 1], tsne_japanese[:, 1]), tsne_orange[:, 1])
        df_tsne['y'] = np.append(np.append(arr_blenheim, arr_japanese), arr_orange)

        df_pca.to_pickle("df_pca.pkl")
        df_tsne.to_pickle("df_tsne.pkl")

    else:
        data_dir_blenheim = pathlib.Path("./dogs_train/Blenheim_spaniel")
        list_blenheim = np.array([item for item in data_dir_blenheim.glob('*')])

        data_dir_japanese = pathlib.Path("./dogs_train/Japanese_spaniel")
        list_japanese = np.array([item for item in data_dir_japanese.glob('*')])

        pca_blenheim, tsne_blenheim = calculate_pca_tsne(list_blenheim)
        pca_japanese, tsne_japanese = calculate_pca_tsne(list_japanese)

        df_pca = pd.DataFrame()
        df_pca['pca-one'] = np.append(pca_blenheim[:, 0], pca_japanese[:, 0])
        df_pca['pca-two'] = np.append(pca_blenheim[:, 1], pca_japanese[:, 1])
        df_pca['pca-three'] = np.append(pca_blenheim[:, 2], pca_japanese[:, 2])
        arr_blenheim = [1 for i in range(len(pca_blenheim))]
        arr_japanese = [2 for i in range(len(pca_japanese))]
        df_pca['y'] = np.append(arr_blenheim, arr_japanese)

        df_tsne = pd.DataFrame()
        df_tsne['tsne-2d-one'] = np.append(tsne_blenheim[:, 0], tsne_japanese[:, 0])
        df_tsne['tsne-2d-two'] = np.append(tsne_blenheim[:, 1], tsne_japanese[:, 1])
        df_tsne['y'] = np.append(arr_blenheim, arr_japanese)

        df_pca.to_pickle("df_pca_non-separable.pkl")
        df_tsne.to_pickle("df_tsne-non-separable.pkl")