"""Benchmarks for KMeans."""
from typing import Any

import numpy as np

from skfda.ml.clustering import KMeans
from skfda import datasets
import matplotlib



class TimeKMeans:
    """Performance of :class:`skfda.ml.clustering.KMeans` for FDataGrid."""

    def setup(self) -> None:
        """Create the data for the test."""
        X, y = datasets.fetch_weather(return_X_y=True, as_frame=True)
        fd = X.iloc[:, 0].values
        fd_temperatures = fd.coordinates[0]
        target = y.values

        indices_samples = np.array([1, 3, 5, 10, 14, 17, 21, 25, 27, 30])
        self.fd = fd_temperatures[indices_samples]


        climates = target[indices_samples].remove_unused_categories()

        colormap = matplotlib.colormaps['tab20b']
        n_climates = len(climates.categories)
        climate_colors = colormap(np.arange(n_climates) / (n_climates - 1))

        n_clusters = n_climates
        seed = 2

        self.kmeans = KMeans(n_clusters=n_clusters, random_state=seed)

    def time_fit_fdgrid(self) -> None:
        """Time to fit KMeans to a FDataGrid."""
        self.kmeans.fit(self.fd)
