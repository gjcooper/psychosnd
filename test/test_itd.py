import numpy as np
import pytest
from psychosnd.itd import itd_diffraction, itd_woodworth


HEAD_RADIUS = 0.0875
C = 343.0


def test_itd_diffraction_zero_angle():
    """ITD at 0° azimuth should be zero (source directly ahead)."""
    result = itd_diffraction([1000.0], [0.0])
    assert result.shape == (1, 1)
    assert abs(result[0, 0]) < 1e-10


def test_itd_diffraction_symmetry():
    """ITD should be antisymmetric: itd(θ) == -itd(-θ)."""
    angles = [30.0, 60.0, 90.0]
    result = itd_diffraction([500.0, 2000.0], angles + [-a for a in angles])
    n = len(angles)
    np.testing.assert_allclose(result[:, :n], -result[:, n:], atol=1e-12)


def test_itd_diffraction_output_shape():
    freqs = [500.0, 1000.0, 4000.0]
    angles = [0.0, 30.0, 60.0, 90.0]
    result = itd_diffraction(freqs, angles)
    assert result.shape == (3, 4)


def test_itd_diffraction_high_freq_approaches_woodworth():
    """At 20 kHz the diffraction ITD should be within ~20 µs of Woodworth (per Fig. 1)."""
    angles = np.linspace(0, 90, 10)
    woodworth = itd_woodworth(angles)
    diffraction = itd_diffraction([20000.0], angles)[0]
    np.testing.assert_allclose(diffraction, woodworth, atol=20e-6)


def test_woodworth_zero_angle():
    assert itd_woodworth([0.0])[0] == pytest.approx(0.0)


def test_woodworth_90_degrees():
    """At 90° the Woodworth ITD should be (a/c)(π/2 + 1)."""
    expected = (HEAD_RADIUS / C) * (np.pi / 2 + 1.0)
    result = itd_woodworth([90.0])[0]
    assert result == pytest.approx(expected)


def test_woodworth_antisymmetry():
    angles = np.array([10.0, 45.0, 90.0, 120.0])
    np.testing.assert_allclose(itd_woodworth(angles), -itd_woodworth(-angles))
