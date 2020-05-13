import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import os
import copy

from sklearn import decomposition, cluster

import umap
import umap.plot


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('--n_dims', type=int, default=25)
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--output_dir', type=str, default='./PCA_output')

    args = parser.parse_args()

    X = np.load(args.data_dir, 'r')

    X = copy.copy(X.reshape(-1, X.shape[-1]))

    ###################### PCA ################################

    X /= np.max(X)

    pca = decomposition.PCA(n_components=args.n_dims)
    pca.fit(X)

    X = pca.transform(X)

    ###################### UMAP ################################

    # mapper = umap.UMAP(n_neighbors=15, n_components=2, metric='correlation', min_dist=0.1).fit(X)
    # umap.plot.points(mapper, labels=Y)
    # umap.plot.plt.show()

    ###################### clustering ################################

    clusters = cluster.KMeans().fit(X)
    print(clusters)
