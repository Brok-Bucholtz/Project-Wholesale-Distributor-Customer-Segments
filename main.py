import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from urllib import urlretrieve
from os.path import isfile
from sklearn.decomposition import PCA, FastICA
from sklearn.mixture import GMM

DATA_FILE_PATH = 'wholesale-customers.csv'
DATA_FILE_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00292/Wholesale%20customers%20data.csv'


def graph_clusters(clusters):
    # Plot the decision boundary by building a mesh grid to populate a graph.
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    hx = (x_max - x_min) / 1000.
    hy = (y_max - y_min) / 1000.
    xx, yy = np.meshgrid(np.arange(x_min, x_max, hx), np.arange(y_min, y_max, hy))

    # Obtain labels for each point in mesh. Use last trained model.
    Z = clusters.predict(np.c_[xx.ravel(), yy.ravel()])
    centroids = clusters.means_

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Paired,
               aspect='auto', origin='lower')

    plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=169, linewidths=3,
                color='w', zorder=10)
    plt.title('Clustering on the wholesale grocery dataset (PCA-reduced data)\n'
              'Centroids are marked with white cross')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()


# Download dataset
if not isfile(DATA_FILE_PATH):
    urlretrieve(DATA_FILE_URL, DATA_FILE_PATH)

# Read dataset
data = pd.read_csv(DATA_FILE_PATH)

# Print the components and the amount of variance in the data contained in each dimension
pca = PCA().fit(data)
pca_component_labels=["PC-"+str(i) for i in range(1, len(pca.components_)+1)]
print '============================'
print 'PCA'
print '============================'
print pd.DataFrame(pca.components_,columns=data.columns,index=pca_component_labels)
print pd.DataFrame(pca.explained_variance_ratio_, columns=['explained_variance_ratio_'], index=pca_component_labels)

# Print the independent components
ica = FastICA().fit(data)
ica_component_labels=['PC-'+str(i) for i in range(1, len(ica.components_)+1)]
print '\n============================'
print 'ICA'
print '============================'
print pd.DataFrame(ica.components_, columns=data.columns, index=ica_component_labels)

# Graph Clusters for size 3 and 4
reduced_data = PCA(2).fit_transform(data)
graph_clusters(GMM(3).fit(reduced_data))
graph_clusters(GMM(4).fit(reduced_data))
