"""Tests of Basis Based Distances."""

import numpy as np
import pytest

from skfda.misc.metrics import BasisBasedDistance
from skfda.representation.basis import FDataBasis, FourierBasis

random_state = np.random.RandomState(23486974)


def test_basis_based_distance_identity() -> None:
    """
    Test that the distance between a function and itself is zero.

    Uses a Fourier basis and a randomly generated function.
    """
    basis = FourierBasis(domain_range=(0, 1), n_basis=5)
    coefs = random_state.rand(3, basis.n_basis)
    fd = FDataBasis(basis, coefs)

    dist = BasisBasedDistance()
    result = dist(fd, fd)
    np.testing.assert_allclose(result, np.zeros(3), atol=1e-12)


def test_basis_based_distance_symmetric() -> None:
    """
    Test that the distance metric is symmetric.

    That is, distance(fd1, fd2) == distance(fd2, fd1).
    """
    basis = FourierBasis(domain_range=(0, 1), n_basis=5)
    fd1 = FDataBasis(basis, random_state.rand(3, 5))
    fd2 = FDataBasis(basis, random_state.rand(3, 5))

    dist = BasisBasedDistance()
    d1 = dist(fd1, fd2)
    d2 = dist(fd2, fd1)

    np.testing.assert_allclose(d1, d2, atol=1e-12)

def test_basis_based_distance_weighted_identity() -> None:
    """
    Test that the weighted distance between a function and itself is zero.

    Uses a Fourier basis and a randomly generated function.
    """
    basis = FourierBasis(domain_range=(0, 1), n_basis=5)
    coefs = random_state.rand(3, basis.n_basis)
    fd = FDataBasis(basis, coefs)

    dist = BasisBasedDistance(random_state.rand(basis.n_basis))
    result = dist(fd, fd)
    np.testing.assert_allclose(result, np.zeros(3), atol=1e-12)


def test_basis_based_distance_weighted_symmetric() -> None:
    """
    Test that the weighted distance metric is symmetric.

    That is, distance(fd1, fd2) == distance(fd2, fd1).
    """
    basis = FourierBasis(domain_range=(0, 1), n_basis=5)
    fd1 = FDataBasis(basis, random_state.rand(3, 5))
    fd2 = FDataBasis(basis, random_state.rand(3, 5))

    dist = BasisBasedDistance(random_state.rand(basis.n_basis))
    d1 = dist(fd1, fd2)
    d2 = dist(fd2, fd1)

    np.testing.assert_allclose(d1, d2, atol=1e-12)


def test_basis_based_distance_weighted() -> None:
    """
    Test distance computation using custom weights.

    Uses two functions with known coefficient differences and equal weights.
    """
    basis = FourierBasis(domain_range=(0, 1), n_basis=5)
    fd1 = FDataBasis(basis, np.array([[1, 2, 3, 4, 5]]))
    fd2 = FDataBasis(basis, np.array([[2, 3, 1, 5, 4]]))
    weights = np.ones(5)

    dist = BasisBasedDistance(weights=weights)
    d = dist(fd1, fd2)

    assert isinstance(d, float)
    assert d > 0
