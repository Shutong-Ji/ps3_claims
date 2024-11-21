import numpy as np
import pytest

from ps3.preprocessing import Winsorizer

# TODO: Test your implementation of a simple Winsorizer
@pytest.mark.parametrize(
    "lower_quantile, upper_quantile", [(0, 1), (0.05, 0.95), (0.5, 0.5)]
)
def test_winsorizer(lower_quantile, upper_quantile):

    X = np.random.normal(0, 1, 1000)

    winsorizer = Winsorizer(lower_quantile = lower_quantile, upper_quantile = upper_quantile)
    winsorizer.fit(X)
    X_clipped = winsorizer.transform(X)
    assert np.all(X_clipped >= np.percentile(X, lower_quantile * 100))
    assert np.all(X_clipped <= np.percentile(X, upper_quantile * 100))
