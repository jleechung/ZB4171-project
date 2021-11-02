import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import Dataset, DataLoader
import warnings

class SpatialDataset(Dataset):
    """ Spatial transcriptomics dataset """

    def __init__(self, counts_path, centroids_path, metadata_path=None,
                 n_neighbors=20, weighted=False, label=None):
        """
        Args:
            counts_path (str): Path to csv file with cell counts
            centroids_path (str): Path to csv file with cell centroids
            metadata_path (str): Path to csv file with metadata
            n_neighbors (int): Number of spatial neighbors
            weighted (bool): Weight neighbors by distance
            label (str): Name of cell class column in metadata file
        """
        self.counts = pd.read_csv(counts_path).to_numpy()
        self.centroids = pd.read_csv(centroids_path).to_numpy()
        assert self.centroids.shape[0] == self.counts.shape[0], \
                'Different number of rows in counts and centroids files'
        self.n_cells = self.counts.shape[0]
        self.n_features = self.counts.shape[1]
        self.n_dims = self.centroids.shape[1]
        self.n_neighbors = n_neighbors
        self.neighbors = NearestNeighbors(n_neighbors = self.n_neighbors + 1)
        self.neighbors.fit(self.centroids)
        self.weighted = weighted
        self.metadata = None
        self.labels = None
        if metadata_path is not None:
            self.metadata = pd.read_csv(metadata_path)
            if label is not None:
                assert label in list(self.metadata), '{} not found in metadata'.format(label)
                self.labels = pd.get_dummies(self.metadata[label]).to_numpy()

    def __repr__(self):
        return "Spatial dataset with {} cells and {} features in {} dimensions".format(self.n_cells, self.n_features, self.n_dims)

    def __len__(self):
        return self.n_cells

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        ## Get current cell, knn
        cell_loc = self.centroids[idx].reshape(1, -1)
        cell_nn = self.neighbors.kneighbors(cell_loc)

        ## Sort nearest neighbors based on angle with index cell
        nn_idx, nn_dist = cell_nn[1].reshape(-1), cell_nn[0].reshape(-1)
        nn_loc = self.centroids[nn_idx]
        nn_loc -= nn_loc[0]
        nn_theta = np.array([self.get_angle(pt) for pt in nn_loc])
        nn_sorted = np.argsort(nn_theta)
        nn_idx, nn_dist = nn_idx[nn_sorted], nn_dist[nn_sorted]

        cell_count = self.counts[nn_idx[0]].reshape(1,-1)
        nn_count = self.counts[nn_idx[1:]]

        ## Weight nearest neighbors based on distance with index cell
        if self.weighted:
            weights = np.sum(nn_dist[1:]) / nn_dist[1:]
            weights /= np.mean(weights)
            nn_count = (nn_count.T * weights).T

        ## Get one hot encoding of cell class
        cell_label, nn_label = None, None
        if self.labels is not None:
            cell_label = self.labels[nn_idx[0]].reshape(1,-1)
            nn_label = self.labels[nn_idx[1:]]

        sample = {'cell_counts': cell_count,
                  'neighbor_counts': nn_count,
                  'cell_labels': cell_label,
                  'neighbor_labels': nn_label}
        return sample

    @staticmethod
    def get_angle(pt):
        x, y = pt[0], pt[1]
        if x == 0 and y == 0:
            return 0
        if x == 0:
            return np.pi/2
        angle = np.arctan(y/x)
        if angle < 0:
            angle += np.pi
        return angle

    def plot(self, pt_size = 1, pt_color = '#1f77b4'):
        plt.scatter(self.centroids[:,0], self.centroids[:,1],
                    s = pt_size, c = pt_color)
        plt.show()
