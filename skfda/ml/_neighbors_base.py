"""Base classes for the neighbor estimators."""
from __future__ import annotations

import copy
from typing import Any, Callable, Generic, Tuple, TypeVar, Union, overload

import numpy as np
import sklearn.neighbors
from scipy.sparse import csr_matrix
from sklearn.utils.validation import check_is_fitted as sklearn_check_is_fitted
from typing_extensions import Literal

from skfda.misc.metrics._utils import PairwiseMetric

from .. import FData, FDataGrid
from .._utils._sklearn_adapter import (
    BaseEstimator,
    ClassifierMixin,
    RegressorMixin,
)
from ..misc.metrics import l2_distance
from ..misc.metrics._typing import Metric
from ..misc.metrics._utils import _fit_metric
from ..representation._typing import NDArrayFloat, NDArrayInt

FDataType = TypeVar("FDataType", bound="FData")
SelfType = TypeVar("SelfType", bound="NeighborsBase[Any, Any]")
SelfTypeRegressor = TypeVar(
    "SelfTypeRegressor",
    bound="NeighborsRegressorMixin[Any, Any]",
)
Input = TypeVar("Input", contravariant=True, bound=Union[NDArrayFloat, FData])
Target = TypeVar("Target")
TargetClassification = TypeVar("TargetClassification", bound=NDArrayInt)
TargetRegression = TypeVar(
    "TargetRegression",
    bound=Union[NDArrayFloat, FData],
)
TargetRegressionMultivariate = TypeVar(
    "TargetRegressionMultivariate",
    bound=NDArrayFloat,
)
TargetRegressionFData = TypeVar(
    "TargetRegressionFData",
    bound=FData,
)

WeightsType = Union[
    Literal["uniform", "distance"],
    Callable[[NDArrayFloat], NDArrayFloat],
]
AlgorithmType = Literal["auto", "ball_tree", "kd_tree", "brute"]


class NeighborsBase(BaseEstimator, Generic[Input, Target]):
    """Base class for nearest neighbors estimators."""

    def __init__(
        self,
        n_neighbors: int | None = None,
        radius: float | None = None,
        weights: WeightsType = "uniform",
        algorithm: AlgorithmType = "auto",
        leaf_size: int = 30,
        metric: Literal["precomputed"] | Metric[Input] = l2_distance,
        n_jobs: int | None = None,
    ):
        self.n_neighbors = n_neighbors
        self.radius = radius
        self.weights = weights
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.metric = metric
        self.n_jobs = n_jobs

    def _check_is_fitted(self) -> None:
        """
        Check if the estimator is fitted.

        Raises:
            NotFittedError: If the estimator is not fitted.

        """
        sklearn_check_is_fitted(self, ['_estimator'])

    def fit(
        self: SelfType,
        X: Input,
        y: Target,
    ) -> SelfType:
        """
        Fit the model using X as training data and y as target values.

        Args:
            X: Training data. FDataGrid with the training data or array matrix
                with shape [n_samples, n_samples] if metric='precomputed'.
            y: Target values of shape = [n_samples] or [n_samples, n_outputs].
                In the case of unsupervised search, this parameter is ignored.

        Returns:
            Self.

        Note:
            This method wraps the corresponding sklearn routine in the module
            ``sklearn.neighbors``.

        """
        return self._fit(X, y)

    def _fit(
        self: SelfType,
        X: Input,
        y: Target,
        fit_with_zeros: bool = True,
    ) -> SelfType:
        # If metric is precomputed no diferences with the Sklearn estimator
        self._estimator = self._init_estimator()

        self._fitted_with_distances = True

        if self.metric == 'precomputed':
            if isinstance(y, FData):  # For functional response regression
                self._fit_y: Target = copy.deepcopy(y)
            self._estimator.fit(X, y)
        else:
            _fit_metric(self.metric, X)
            self._fit_X = copy.deepcopy(X)
            self._fit_y = copy.deepcopy(y)
            if fit_with_zeros:
                self._fitted_with_distances = False
                self._estimator.fit(np.zeros(shape=(len(X), len(X))), y)
            else:
                distances = PairwiseMetric(self.metric)(X)
                self._estimator.fit(distances, y)

        return self

    def _refit_with_distances(self) -> None:
        if not self._fitted_with_distances:
            assert self.metric != "precomputed"
            distances = PairwiseMetric(self.metric)(self._fit_X)
            self._estimator.fit(distances, self._fit_y)
            self._fitted_with_distances = True

    def _X_to_distances(
        self,
        X: Input,
    ) -> NDArrayFloat:

        if self.metric == 'precomputed':
            return X

        return PairwiseMetric(self.metric)(X, self._fit_X)

    def _init_estimator(
        self,
    ) -> sklearn.neighbors.NearestNeighbors:
        """Initialize the sklearn nearest neighbors estimator."""
        return sklearn.neighbors.NearestNeighbors(
            n_neighbors=self.n_neighbors,
            radius=self.radius,
            algorithm=self.algorithm,
            leaf_size=self.leaf_size,
            metric="precomputed",
            n_jobs=self.n_jobs,
        )


class KNeighborsMixin(NeighborsBase[Input, Target]):
    """Mixin class for K-Neighbors."""

    @overload
    def kneighbors(
        self,
        X: Input | None = None,
        n_neighbors: int | None = None,
        *,
        return_distance: Literal[True] = True,
    ) -> Tuple[NDArrayFloat, NDArrayInt]:
        pass

    @overload
    def kneighbors(
        self,
        X: Input | None = None,
        n_neighbors: int | None = None,
        *,
        return_distance: Literal[False],
    ) -> NDArrayInt:
        pass

    def kneighbors(
        self,
        X: Input | None = None,
        n_neighbors: int | None = None,
        *,
        return_distance: bool = True,
    ) -> NDArrayInt | Tuple[NDArrayFloat, NDArrayInt]:
        """
        Find the K-neighbors of a point.

        Returns indices of and distances to the neighbors of each point.

        Args:
            X: FDatagrid with the query functions or  matrix
                (n_query, n_indexed) if metric == 'precomputed'. If not
                provided, neighbors of each indexed point are returned. In
                this case, the query point is not considered its own neighbor.
            n_neighbors: Number of neighbors to get (default is the value
                passed to the constructor).
            return_distance: Defaults to True. If False,
                distances will not be returned.

        Returns:
            dist : array
                Array representing the lengths to points, only present if
                return_distance=True
            ind : array
                Indices of the nearest points in the population matrix.

        Examples:
            Firstly, we will create a toy dataset.

            >>> from skfda.datasets import make_sinusoidal_process
            >>> fd1 = make_sinusoidal_process(phase_std=.25, random_state=0)
            >>> fd2 = make_sinusoidal_process(phase_mean=1.8, error_std=0.,
            ...                               phase_std=.25, random_state=0)
            >>> fd = fd1.concatenate(fd2)

            We will fit a Nearest Neighbors estimator

            >>> from skfda.ml.clustering import NearestNeighbors
            >>> neigh = NearestNeighbors()
            >>> neigh.fit(fd)
            NearestNeighbors(...)

            Now we can query the k-nearest neighbors.

            >>> distances, index = neigh.kneighbors(fd[:2])
            >>> index # Index of k-neighbors of samples 0 and 1
            array([[ 0,  7,  6, 11,  2],...)

            >>> distances.round(2) # Distances to k-neighbors
            array([[ 0.  ,  0.28,  0.29,  0.29,  0.3 ],
                   [ 0.  ,  0.27,  0.28,  0.29,  0.3 ]])

        Notes:
            This method wraps the corresponding sklearn routine in the
            module ``sklearn.neighbors``.

        """
        self._check_is_fitted()
        if X is None:
            self._refit_with_distances()

        X_dist = None if X is None else self._X_to_distances(X)

        return self._estimator.kneighbors(  # type: ignore [no-any-return]
            X_dist,
            n_neighbors,
            return_distance,
        )

    def kneighbors_graph(
        self,
        X: Input | None = None,
        n_neighbors: int | None = None,
        mode: Literal["connectivity", "distance"] = "connectivity",
    ) -> csr_matrix:
        """
        Compute the (weighted) graph of k-Neighbors for points in X.

        Args:
            X: FDatagrid with the query functions or  matrix
                (n_query, n_indexed) if metric == 'precomputed'. If not
                provided, neighbors of each indexed point are returned. In
                this case, the query point is not considered its own neighbor.
            n_neighbors: Number of neighbors to get (default is the value
                passed to the constructor).
            mode: Type of returned matrix: 'connectivity' will return the
                connectivity matrix with ones and zeros, in 'distance' the
                edges are distance between points.

        Returns:
            Sparse matrix in CSR format, shape = [n_samples, n_samples_fit]
            n_samples_fit is the number of samples in the fitted data
            A[i, j] is assigned the weight of edge that connects i to j.

        Examples:
            Firstly, we will create a toy dataset.

            >>> from skfda.datasets import make_sinusoidal_process
            >>> fd1 = make_sinusoidal_process(phase_std=.25, random_state=0)
            >>> fd2 = make_sinusoidal_process(phase_mean=1.8, error_std=0.,
            ...                               phase_std=.25, random_state=0)
            >>> fd = fd1.concatenate(fd2)

            We will fit a Nearest Neighbors estimator.

            >>> from skfda.ml.clustering import NearestNeighbors
            >>> neigh = NearestNeighbors()
            >>> neigh.fit(fd)
            NearestNeighbors(...)

            Now we can obtain the graph of k-neighbors of a sample.

            >>> graph = neigh.kneighbors_graph(fd[0])
            >>> print(graph)
              (0, 0)	1.0
              (0, 7)	1.0
              (0, 6)	1.0
              (0, 11)	1.0
              (0, 2)	1.0

        Notes:
            This method wraps the corresponding sklearn routine in the
            module ``sklearn.neighbors``.

        """
        self._check_is_fitted()
        if X is None:
            self._refit_with_distances()

        X_dist = None if X is None else self._X_to_distances(X)

        return self._estimator.kneighbors_graph(X_dist, n_neighbors, mode)


class RadiusNeighborsMixin(NeighborsBase[Input, Target]):
    """Mixin Class for Raius Neighbors."""

    @overload
    def radius_neighbors(
        self,
        X: Input | None = None,
        radius: float | None = None,
        *,
        return_distance: Literal[True] = True,
    ) -> Tuple[NDArrayFloat, NDArrayInt]:  # TODO: Fix return type
        pass

    @overload
    def radius_neighbors(
        self,
        X: Input | None = None,
        radius: float | None = None,
        *,
        return_distance: Literal[False],
    ) -> NDArrayInt:
        pass

    def radius_neighbors(
        self,
        X: Input | None = None,
        radius: float | None = None,
        *,
        return_distance: bool = True,
    ) -> NDArrayInt | Tuple[NDArrayFloat, NDArrayInt]:  # TODO: Fix return type
        """
        Find the neighbors within a given radius of a fdatagrid.

        Return the indices and distances of each point from the dataset
        lying in a ball with size ``radius`` around the points of the query
        array. Points lying on the boundary are included in the results.
        The result points are *not* necessarily sorted by distance to their
        query point.

        Args:
            X: Sample or samples whose neighbors will be returned. If not
                provided, neighbors of each indexed point are returned. In this
                case, the query point is not considered its own neighbor.
            radius: Limiting distance of neighbors to return.
                (default is the value passed to the constructor).
            return_distance: Defaults to True. If False, distances will not be
                returned.

        Returns:
            (array, shape (n_samples): dist : array of arrays representing the
                distances to each point, only present if return_distance=True.
                The distance values are computed according to the ``metric``
                constructor parameter.
            (array, shape (n_samples,): An array of arrays of indices of the
                approximate nearest points from the population matrix that lie
                within a ball of size ``radius`` around the query points.

        Examples:
            Firstly, we will create a toy dataset.

            >>> from skfda.datasets import make_sinusoidal_process
            >>> fd1 = make_sinusoidal_process(phase_std=.25, random_state=0)
            >>> fd2 = make_sinusoidal_process(phase_mean=1.8, error_std=0.,
            ...                               phase_std=.25, random_state=0)
            >>> fd = fd1.concatenate(fd2)

            We will fit a Nearest Neighbors estimator.

            >>> from skfda.ml.clustering import NearestNeighbors
            >>> neigh = NearestNeighbors(radius=.3)
            >>> neigh.fit(fd)
            NearestNeighbors(...radius=0.3...)

            Now we can query the neighbors in the radius.

            >>> distances, index = neigh.radius_neighbors(fd[:2])
            >>> index[0] # Neighbors of sample 0
            array([ 0,  2,  6,  7, 11]...)

            >>> distances[0].round(2) # Distances to neighbors of the sample 0
            array([ 0.  ,  0.3 ,  0.29,  0.28,  0.29])


        See also:
            kneighbors

        Notes:
            Because the number of neighbors of each point is not necessarily
            equal, the results for multiple query points cannot be fit in a
            standard data array.
            For efficiency, `radius_neighbors` returns arrays of objects, where
            each object is a 1D array of indices or distances.

            This method wraps the corresponding sklearn routine in the module
            ``sklearn.neighbors``.

        """
        self._check_is_fitted()
        if X is None:
            self._refit_with_distances()

        X_dist = None if X is None else self._X_to_distances(X)

        return (  # type: ignore [no-any-return]
            self._estimator.radius_neighbors(
                X_dist,
                radius=radius,
                return_distance=return_distance,
            )
        )

    def radius_neighbors_graph(
        self,
        X: Input | None = None,
        radius: float | None = None,
        mode: Literal["connectivity", "distance"] = 'connectivity',
    ) -> csr_matrix:
        """
        Compute the (weighted) graph of Neighbors for points in X.

        Neighborhoods are restricted the points at a distance lower than
        radius.

        Args:
            X:  The query sample or samples. If not provided, neighbors of
                each indexed point are returned. In this case, the query
                point is not considered its own neighbor.
            radius: Radius of neighborhoods. (default is the value passed
                to the constructor).
            mode: Type of returned matrix: 'connectivity' will return the
                connectivity matrix with ones and zeros, in 'distance'
                the edges are distance between points.

        Returns:
            sparse matrix in CSR format, shape = [n_samples, n_samples]
            A[i, j] is assigned the weight of edge that connects i to j.

        Notes:
            This method wraps the corresponding sklearn routine in the module
            ``sklearn.neighbors``.

        """
        self._check_is_fitted()
        if X is None:
            self._refit_with_distances()

        X_dist = None if X is None else self._X_to_distances(X)

        return self._estimator.radius_neighbors_graph(
            X_dist,
            radius=radius,
            mode=mode,
        )


class NeighborsClassifierMixin(
    NeighborsBase[Input, Target],
    ClassifierMixin[Input, Target],
):
    """Mixin class for classifiers based in nearest neighbors."""

    def predict(
        self,
        X: Input,
    ) -> Target:
        """
        Predict the class labels for the provided data.

        Args:
            X: Test samples or array (n_query, n_indexed) if metric ==
                'precomputed'.

        Returns:
            Array of shape [n_samples] or [n_samples, n_outputs] with class
            labels for each data sample.

        Notes:
            This method wraps the corresponding sklearn routine in the module
            ``sklearn.neighbors``.

        """
        self._check_is_fitted()

        X_dist = self._X_to_distances(X)

        return self._estimator.predict(X_dist)  # type: ignore [no-any-return]

    def predict_proba(
        self,
        X: Input,
    ) -> NDArrayFloat:
        """
        Calculate probability estimates for the test data X.

        Args:
            X: FDataGrid with the test samples or array (n_query, n_indexed)
                if metric == 'precomputed'.

        Returns:
            The class probabilities of the input samples. Classes are
            ordered by lexicographic order.

        """
        self._check_is_fitted()

        X_dist = self._X_to_distances(X)

        return (  # type: ignore [no-any-return]
            self._estimator.predict_proba(X_dist)
        )


class NeighborsRegressorMixin(
    NeighborsBase[Input, TargetRegression],
    RegressorMixin[Input, TargetRegression],
):
    """Mixin class for the regressors based on neighbors."""

    def _average(
        self,
        X: TargetRegression,
        weights: NDArrayFloat | None = None,
    ) -> TargetRegression:
        """Compute weighted average."""
        if weights is None:
            return np.mean(X, axis=0)  # type: ignore [no-any-return]

        weights /= np.sum(weights)

        return np.sum(X * weights, axis=0)  # type: ignore [no-any-return]

    def _prediction_from_neighbors(
        self,
        neighbors: TargetRegression,
        distance: NDArrayFloat,
    ) -> TargetRegression:

        if self.weights == 'uniform':
            weights = None
        elif self.weights == 'distance':
            weights = self._distance_weights(distance)
        else:
            weights = self.weights(distance)

        return self._average(neighbors, weights)

    def fit(
        self: SelfTypeRegressor,
        X: Input,
        y: TargetRegression,
    ) -> SelfTypeRegressor:
        """
        Fit the model using X as training data and y as responses.

        Args:
            X: Training data. FDataGrid
                with the training data or array matrix with shape
                [n_samples, n_samples] if metric='precomputed'.
            y: Training data. FData with the training respones (functional
                response case) or array matrix with length `n_samples` in
                the multivariate response case.

        Returns:
            Self.

        """
        self._functional = isinstance(y, FData)
        return super().fit(X, y)

    def _distance_weights(
        self,
        distance: NDArrayFloat,
    ) -> NDArrayFloat:
        """Return weights based on distance reciprocal."""
        idx = (distance == 0)
        if np.any(idx):
            weights = distance
            weights[idx] = 1
            weights[~idx] = 0
        else:
            weights = 1 / distance

        return weights

    def predict(
        self,
        X: Input,
    ) -> TargetRegression:
        """
        Predict the target for the provided data.

        Args:
            X: FDataGrid with the test
                samples or array (n_query, n_indexed) if metric ==
                'precomputed'.

        Returns:
            array of shape = [n_samples] or [n_samples, n_outputs]
            or :class:`FData` containing as many samples as X.

        """
        self._check_is_fitted()

        # Choose type of prediction
        if self._functional:
            return self._functional_predict(X)

        return self._multivariate_predict(X)

    def _multivariate_predict(
        self: NeighborsRegressorMixin[Input, TargetRegressionMultivariate],
        X: Input,
    ) -> TargetRegressionMultivariate:
        """Predict a multivariate target."""
        X_dist = self._X_to_distances(X)

        return self._estimator.predict(X_dist)  # type: ignore [no-any-return]

    def _functional_predict(
        self: NeighborsRegressorMixin[FData, Any],
        X: Input,
    ) -> TargetRegression:
        """Predict functional responses."""
        distances, neighbors = self._query(X)

        if len(neighbors[0]) == 0:
            pred = self._outlier_response(neighbors)
        else:
            pred = self._prediction_from_neighbors(
                self._fit_y[neighbors[0]],
                distances[0],
            )

        for i, idx in enumerate(neighbors[1:]):
            if len(idx) == 0:
                new_pred = self._outlier_response(neighbors)
            else:
                new_pred = self._prediction_from_neighbors(
                    self._fit_y[idx],
                    distances[i + 1],
                )

            pred = pred.concatenate(new_pred)

        return pred

    def _outlier_response(
        self,
        neighbors: TargetRegression,
    ) -> TargetRegression:
        """Response in case of no neighbors."""
        outlier_response = getattr(self, "outlier_response", None)

        if outlier_response is None:
            index = np.where([len(n) == 0 for n in neighbors])[0]

            raise ValueError(
                f"No neighbors found for test samples  {index}, "
                "you can try using larger radius, give a reponse "
                "for outliers, or consider removing them from "
                "your dataset.",
            )

        return outlier_response

    def score(
        self,
        X: Input,
        y: TargetRegression,
        sample_weight: NDArrayFloat | None = None,
    ) -> float:
        r"""Return the coefficient of determination R^2 of the prediction.

        In the multivariate response case, the coefficient :math:`R^2` is
        defined as

        .. math::
            1 - \frac{\sum_{i=1}^{n} (y_i - \hat y_i)^2}
            {\sum_{i=1}^{n} (y_i - \frac{1}{n}\sum_{i=1}^{n}y_i)^2}

        where :math:`\hat{y}_i` is the prediction associated to the test sample
        :math:`X_i`, and :math:`{y}_i` is the true response. See
        :func:`sklearn.metrics.r2_score <sklearn.metrics.r2_score>` for more
        information.


        In the functional case it is returned an extension of the coefficient
        of determination :math:`R^2`, defined as

        .. math::
            1 - \frac{\sum_{i=1}^{n}\int (y_i(t) - \hat{y}_i(t))^2dt}
            {\sum_{i=1}^{n} \int (y_i(t)- \frac{1}{n}\sum_{i=1}^{n}y_i(t))^2dt}


        The best possible score is 1.0 and it can be negative
        (because the model can be arbitrarily worse). A constant model that
        always predicts the expected value of y, disregarding the input
        features, would get a R^2 score of 0.0.

        Args:
            X: Test samples to be predicted.
            y: True responses of the test samples.
            sample_weight: Sample weights.

        Returns:
            Coefficient of determination.

        """
        if self._functional:
            return self._functional_score(X, y, sample_weight=sample_weight)

        # Default sklearn multivariate score
        return super().score(X, y, sample_weight=sample_weight)

    def _functional_score(
        self: NeighborsRegressorMixin[Input, TargetRegressionFData],
        X: Input,
        y: TargetRegressionFData,
        sample_weight: NDArrayFloat | None = None,
    ) -> float:
        r"""
        Return an extension of the coefficient of determination R^2.

        The coefficient is defined as

        .. math::
            1 - \frac{\sum_{i=1}^{n}\int (y_i(t) - \hat{y}_i(t))^2dt}
            {\sum_{i=1}^{n} \int (y_i(t)- \frac{1}{n}\sum_{i=1}^{n}y_i(t))^2dt}

        where :math:`\hat{y}_i` is the prediction associated to the test sample
        :math:`X_i`, and :math:`{y}_i` is the true response.

        The best possible score is 1.0 and it can be negative
        (because the model can be arbitrarily worse). A constant model that
        always predicts the expected value of y, disregarding the input
        features, would get a R^2 score of 0.0.

        Args:
            X: Test samples to be predicted.
            y: True responses of the test samples.
            sample_weight (array_like, shape = [n_samples], optional): Sample
                weights.

        Returns:
            Coefficient of determination.

        """
        # TODO: If it is created a module in ml.regression with other
        # score metrics, move it.
        from scipy.integrate import simps

        if y.dim_codomain != 1 or y.dim_domain != 1:
            raise ValueError(
                "Score not implemented for multivariate "
                "functional data.",
            )

        # Make prediction
        pred = self.predict(X)

        u = y - pred
        v = y - y.mean()

        # Discretize to integrate and make squares if needed
        if type(u) != FDataGrid:
            u = u.to_grid()
            v = v.to_grid()

        data_u = u.data_matrix[..., 0]
        data_v = v.data_matrix[..., 0]

        # Square without allocate more memory
        np.square(data_u, out=data_u)
        np.square(data_v, out=data_v)

        if sample_weight is not None:
            if len(sample_weight) != len(y):
                raise ValueError("Must be a weight for each sample.")

            sample_weight = np.asarray(sample_weight)
            sample_weight = sample_weight / sample_weight.sum()
            data_u_t = data_u.T
            data_u_t *= sample_weight
            data_v_t = data_v.T
            data_v_t *= sample_weight

        # Sum and integrate
        sum_u = np.sum(data_u, axis=0)
        sum_v = np.sum(data_v, axis=0)

        int_u = simps(sum_u, x=u.grid_points[0])
        int_v = simps(sum_v, x=v.grid_points[0])

        return 1 - int_u / int_v
