import numpy as np
import pytest

from src.opticaldisp.waveforms import (
    generate_gaussian,
    generate_lorentzian,
    generate_sech,
    generate_square,
)


@pytest.fixture
def setup_waveform():
    """Fixture to create a time array and set the pulse width of the waveform."""
    t = np.linspace(-5, 5, 1000)  # Time vector
    pulsewidth = 1  # Pulse width for testing
    return t, pulsewidth


def test_generate_gaussian(setup_waveform):
    """Check that the Gaussian waveform has the expected maximum value."""
    t, pulsewidth = setup_waveform
    waveform = generate_gaussian(t, pulsewidth)
    expected_max = np.exp(0)  # Max value of Gaussian at t=0
    assert np.isclose(
        np.max(waveform), expected_max, atol=1e-3
    ), f"Gaussian max: {np.max(waveform)} != {expected_max}"


def test_generate_square(setup_waveform):
    """Check that all the values of the square waveform are either 0 or 1."""
    t, pulsewidth = setup_waveform
    waveform = generate_square(t, pulsewidth)
    assert np.all(
        (waveform == 0) | (waveform == 1)
    ), "Square waveform contains values other than 0 or 1"


def test_generate_lorentzian(setup_waveform):
    """Check that the Lorentzian waveform has the expected maximum value."""
    t, pulsewidth = setup_waveform
    waveform = generate_lorentzian(t, pulsewidth)
    expected_max = 1 / (1 + (0 / (pulsewidth / 1.287)) ** 2)  # Max value at t=0
    assert np.isclose(
        np.max(waveform), expected_max, atol=1e-3
    ), f"Lorentzian max: {np.max(waveform)} != {expected_max}"


def test_generate_sech(setup_waveform):
    """Check that the Sech waveform has the expected maximum value."""
    t, pulsewidth = setup_waveform
    waveform = generate_sech(t, pulsewidth)
    expected_max = 1 / np.cosh(0)  # Max value of Sech at t=0
    assert np.isclose(
        np.max(waveform), expected_max, atol=1e-3
    ), f"Sech max: {np.max(waveform)} != {expected_max}"
