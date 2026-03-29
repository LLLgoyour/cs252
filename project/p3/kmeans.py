'''kmeans.py
Performs K-Means clustering
Jack Dai
CS 252: Mathematical Data Analysis and Visualization
Spring 2026
'''
import numpy as np
import matplotlib.pyplot as plt


class KMeans:
    def __init__(self, data=None):
        '''KMeans constructor

        (Should not require any changes)

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features)
        '''
        # k: int. Number of clusters
        self.k = None
        # centroids: ndarray. shape=(k, self.num_features)
        #   k cluster centers
        self.centroids = None
        # data_centroid_labels: ndarray of ints. shape=(self.num_samps,)
        #   Holds index of the assigned cluster of each data sample
        self.data_centroid_labels = None

        # inertia: float.
        #   Mean squared distance between each data sample and its assigned (nearest) centroid
        self.inertia = None

        # num_samps: int. Number of samples in the dataset
        self.num_samps = None
        # num_features: int. Number of features (variables) in the dataset
        self.num_features = None

        if data is not None:
            # data: ndarray. shape=(num_samps, num_features)
            self.data = data.copy()
            self.num_samps, self.num_features = data.shape

    def set_data(self, data):
        '''Replaces data instance variable with `data`.

        Reminder: Make sure to update the number of data samples and features!

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features)
        '''
        self.data = data.copy()
        self.num_samps, self.num_features = data.shape
        
    def get_data(self):
        '''Get a COPY of the data

        Returns:
        -----------
        ndarray. shape=(num_samps, num_features). COPY of the data
        '''
        return self.data.copy()

    def get_centroids(self):
        '''Get the K-means centroids

        (Should not require any changes)

        Returns:
        -----------
        ndarray. shape=(k, self.num_features).
        '''
        return self.centroids

    def get_data_centroid_labels(self):
        '''Get the data-to-cluster assignments

        (Should not require any changes)

        Returns:
        -----------
        ndarray of ints. shape=(self.num_samps,)
        '''
        return self.data_centroid_labels

    def dist_pt_to_pt(self, pt_1, pt_2):
        '''Compute the Euclidean distance between data samples `pt_1` and `pt_2`

        Parameters:
        -----------
        pt_1: ndarray. shape=(num_features,)
        pt_2: ndarray. shape=(num_features,)

        Returns:
        -----------
        float. Euclidean distance between `pt_1` and `pt_2`.

        NOTE:
        - Implement without any for loops (you will thank yourself later since you will wait
        only a small fraction of the time for your code to stop running).
        - Implement the distance formula (see notebook), do not use np.linalg.norm here.
        '''
        return np.sqrt(np.sum((pt_1 - pt_2) ** 2))

    def dist_pt_to_centroids(self, pt, centroids):
        '''Compute the Euclidean distance between data sample `pt` and and all the cluster centroids
        self.centroids

        Parameters:
        -----------
        pt: ndarray. shape=(num_features,)
        centroids: ndarray. shape=(C, num_features)
            C centroids, where C is an int.

        Returns:
        -----------
        ndarray. shape=(C,).
            distance between pt and each of the C centroids in `centroids`.

        NOTE:
        - Implement without any for loops (you will thank yourself later since you will wait
        only a small fraction of the time for your code to stop running).
        - Implement the distance formula (see notebook), do not use np.linalg.norm here.
        '''
        return np.sqrt(np.sum((centroids - pt) ** 2, axis=1))

    def initialize(self, k):
        '''Initializes K-means by setting the initial centroids (means) to K unique randomly
        selected data samples and inertia to infinity (i.e. np.inf).

        Parameters:
        -----------
        k: int. Number of clusters

        Returns:
        -----------
        ndarray. shape=(k, self.num_features). Initial centroids for the k clusters.

        NOTE: Can be implemented without any for loops
        '''
        if self.data is None or self.num_samps is None:
            raise ValueError('Data must be set before initialization.')
        self.k = k
        rand_inds = np.random.choice(int(self.num_samps), size=k, replace=False)
        self.centroids = self.data[rand_inds].copy()
        self.inertia = np.inf
        return self.centroids

    def cluster(self, k=2, tol=1e-2, max_iter=1000, verbose=False):
        '''Performs K-means clustering on the data

        Parameters:
        -----------
        k: int. Number of clusters
        tol: float. Terminate K-means if the (absolute value of) the absolute difference in the inertia from the
            previous and current time step < `tol`.
        max_iter: int. Make sure that K-means does not run more than `max_iter` iterations.
        verbose: boolean. Print out debug information if set to True.

        Returns:
        -----------
        self.inertia. float. Mean squared distance between each data sample and its cluster mean
        int. Number of iterations that K-means was run for

        TODO:
        1. Initialize K-means variables
        2. Do K-means as long as the max number of iterations is not met AND the absolute value of the difference between
        the previous and current inertia is bigger than the tolerance `tol`. K-means should always run for at least 1
        iteration.
        3. Set instance variables based on computed values.
        (All instance variables defined in constructor should be populated with meaningful values)
        4. Print out total number of iterations K-means ran for
        '''
        if self.data is None:
            raise ValueError('Data must be set before clustering.')
        num_samps = self.data.shape[0]
        centroids = self.initialize(k)
        prev_inertia = np.inf
        cur_inertia = np.inf
        num_iter = 0
        labels = np.zeros(num_samps, dtype=int)

        while num_iter < max_iter and (num_iter == 0 or abs(prev_inertia - cur_inertia) > tol):
            labels = self.update_labels(centroids)
            new_centroids, _ = self.update_centroids(k, labels, centroids)

            self.centroids = new_centroids
            self.data_centroid_labels = labels

            prev_inertia = cur_inertia
            cur_inertia = self.compute_inertia()
            centroids = new_centroids
            num_iter += 1

            if verbose:
                print(f'Iteration {num_iter}, inertia={cur_inertia:.6f}')

        self.k = k
        self.centroids = centroids
        self.data_centroid_labels = labels
        self.inertia = cur_inertia

        print(f'K-means ran for {num_iter} iterations')
        return self.inertia, num_iter

    def cluster_batch(self, k=2, n_iter=1, verbose=False):
        '''Run K-means multiple times, each time with different initial conditions.
        Keeps track of K-means instance that generates lowest inertia. Sets the following instance
        variables based on the best K-mean run:
        - self.centroids
        - self.data_centroid_labels
        - self.inertia

        Parameters:
        -----------
        k: int. Number of clusters
        n_iter: int. Number of times to run K-means with the designated `k` value.
        verbose: boolean. Print out debug information if set to True.
        '''
        if self.data is None:
            raise ValueError('Data must be set before clustering.')
        best_inertia = np.inf
        best_centroids = None
        best_labels = None

        for i in range(n_iter):
            inertia, _ = self.cluster(k=k, verbose=verbose)
            if verbose:
                print(f'Batch run {i+1}/{n_iter} inertia={inertia:.6f}')

            if inertia < best_inertia:
                best_inertia = inertia
                best_centroids = self.centroids.copy() if self.centroids is not None else None
                best_labels = self.data_centroid_labels.copy() if self.data_centroid_labels is not None else None

        if best_centroids is None or best_labels is None:
            raise RuntimeError('K-means batch run did not produce valid centroids/labels.')

        self.k = k
        self.centroids = best_centroids
        self.data_centroid_labels = best_labels
        self.inertia = best_inertia

    def update_labels(self, centroids):
        '''Assigns each data sample to the nearest centroid

        Parameters:
        -----------
        centroids: ndarray. shape=(k, self.num_features). Current centroids for the k clusters.

        Returns:
        -----------
        ndarray of ints. shape=(self.num_samps,). Holds index of the assigned cluster of each data
            sample. These should be ints (pay attention to/cast your dtypes accordingly).

        Example: If we have 3 clusters and we compute distances to data sample i: [0.1, 0.5, 0.05]
        labels[i] is 2. The entire labels array may look something like this: [0, 2, 1, 1, 0, ...]
        '''
        if self.data is None:
            raise ValueError('Data must be set before updating labels.')
        dists = np.sqrt(np.sum((self.data[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2, axis=2))
        return np.argmin(dists, axis=1).astype(int)

    def update_centroids(self, k, data_centroid_labels, prev_centroids):
        '''Computes each of the K centroids (means) based on the data assigned to each cluster

        Parameters:
        -----------
        k: int. Number of clusters
        data_centroid_labels. ndarray of ints. shape=(self.num_samps,)
            Holds index of the assigned cluster of each data sample
        prev_centroids. ndarray. shape=(k, self.num_features)
            Holds centroids for each cluster computed on the PREVIOUS time step

        Returns:
        -----------
        new_centroids. ndarray. shape=(k, self.num_features).
            Centroids for each cluster computed on the CURRENT time step
        centroid_diff. ndarray. shape=(k, self.num_features).
            Difference between current and previous centroid values

        NOTE: Your implementation should handle the case when there are no samples assigned to a cluster —
        i.e. `data_centroid_labels` does not have a valid cluster index in it at all.
            For example, if `k`=3 and data_centroid_labels = [0, 1, 0, 0, 1], there are no samples assigned to cluster 2.
        In the case of each cluster without samples assigned to it, you should assign make its centroid a data sample
        randomly selected from the dataset.
        '''
        if self.data is None or self.num_features is None or self.num_samps is None:
            raise ValueError('Data must be set before updating centroids.')
        new_centroids = np.zeros((k, int(self.num_features)))

        for c in range(k):
            cluster_data = self.data[data_centroid_labels == c]
            if cluster_data.shape[0] == 0:
                rand_ind = np.random.randint(0, int(self.num_samps))
                new_centroids[c] = self.data[rand_ind]
            else:
                new_centroids[c] = np.mean(cluster_data, axis=0)

        centroid_diff = new_centroids - prev_centroids
        return new_centroids, centroid_diff

    def compute_inertia(self):
        '''Mean squared distance between every data sample and its assigned (nearest) centroid

        Returns:
        -----------
        float. The average squared distance between every data sample and its assigned cluster centroid.
        '''
        if self.data is None or self.centroids is None or self.data_centroid_labels is None:
            raise ValueError('Data, centroids, and labels must be set before computing inertia.')
        sq_dists = np.sum((self.data - self.centroids[self.data_centroid_labels]) ** 2, axis=1)
        return np.mean(sq_dists)

    def plot_clusters(self):
        '''Creates a scatter plot of the data color-coded by cluster assignment.

        TODO:
        - Plot samples belonging to a cluster with the same color.
        - Plot the centroids in black with a different plot marker.
        - The default scatter plot color palette produces colors that may be difficult to discern
        (especially for those who are colorblind). To make sure you change your colors to be clearly differentiable,
        use either the Okabe & Ito or one of the Petroff color palettes: https://github.com/proplot-dev/proplot/issues/424
        Each string in the `colors` list that starts with # is the hexadecimal representation of a color (blue, red, etc.)
        that can be passed into the color `c` keyword argument of plt.plot or plt.scatter.
            Pick one of the palettes with a generous number of colors so that you don't run out if k is large (e.g. >6).
        '''
        if self.data is None or self.centroids is None or self.data_centroid_labels is None or self.k is None:
            raise ValueError('Data, centroids, labels, and k must be set before plotting clusters.')
        colors = [
            '#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2',
            '#D55E00', '#CC79A7', '#999999', '#332288', '#88CCEE'
        ]

        for c in range(self.k):
            cluster_data = self.data[self.data_centroid_labels == c]
            if cluster_data.shape[0] > 0:
                plt.scatter(cluster_data[:, 0], cluster_data[:, 1], c=colors[c % len(colors)], s=35)

        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], c='black', marker='X', s=120)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'K-means Clusters (k={self.k})')

    def elbow_plot(self, max_k, n_iter=1):
        '''Makes an elbow plot: cluster number (k) on x axis, inertia on y axis.

        Parameters:
        -----------
        max_k: int. Run k-means with k=1,2,...,max_k.
        n_iter: int. Number of K-means runs to perform per k value (best run is kept).

        TODO:
        - Run k-means with k=1,2,...,max_k, record the inertia.
        - Make the plot with appropriate x label, and y label, x tick marks.
        '''
        ks = np.arange(1, max_k + 1)
        inertias = []

        for k in ks:
            self.cluster_batch(k=k, n_iter=n_iter)
            inertias.append(self.inertia)

        plt.plot(ks, inertias, marker='o')
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('Inertia')
        plt.xticks(ks)
        plt.title('Elbow Plot')

    def replace_color_with_centroid(self):
        '''Replace each RGB pixel in self.data (flattened image) with the closest centroid value.
        Used with image compression after K-means is run on the image vector.

        Parameters:
        -----------
        None

        Returns:
        -----------
        None
        '''
        if self.centroids is None or self.data_centroid_labels is None:
            raise ValueError('Centroids and labels must be set before replacing colors.')
        self.data = self.centroids[self.data_centroid_labels].copy()

    def segment_cluster(self, k):
        '''Segments cluster `k` from a COPY of the data. This means setting the entries in all samples NOT assigned to
        cluster `k` to 0.

        Parameters:
        -----------
        k: int.
            The ID of the cluster to segment.

        Returns:
        --------
        ndarray : shape=(num_samples, num_variables).
            COPY of the dataset containing all 0s except for samples that belong to cluster k, which preserve their
            original data.

        Note: Logical indexing can be helpful here.
        '''
        if self.data is None or self.data_centroid_labels is None:
            raise ValueError('Data and labels must be set before segmenting a cluster.')
        seg_data = self.data.copy()
        seg_data[self.data_centroid_labels != k] = 0
        return seg_data
