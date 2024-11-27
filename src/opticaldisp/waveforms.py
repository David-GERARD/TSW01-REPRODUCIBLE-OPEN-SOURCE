import numpy as np


def generate_gaussian(t, pulsewidth):
    """
    Generates a Gaussian waveform.

    Parameters:
    t (array-like): Input array representing time values.
    pulsewidth (float): The width of the pulse.

    Returns:
    np.ndarray: Gaussian waveform.
    """
    # Calculate tau based on the pulse width
    tau = pulsewidth / (2 * np.sqrt(np.log(2)))
    # Compute the Gaussian waveform
    gaussian = np.exp(-(t**2) / (2 * tau**2))
    return gaussian


def generate_square(t, pulsewidth):
    """
    Generates a square waveform.

    Parameters:
    t (array-like): Input array representing time values.
    pulsewidth (float): The width of the pulse.

    Returns:
    np.ndarray: Square wave with values 1 within pulsewidth and 0 outside.
    """
    # Initialize the square wave array with ones
    square = np.ones_like(t)
    # Set values to 0 where abs(t) is greater than pulsewidth / 2
    square[np.abs(t) > pulsewidth / 2] = 0
    return square


def generate_lorentzian(t, pulsewidth):
    """
    Generates a Lorentzian waveform.

    Parameters:
    t (array-like): Input array representing time values.
    pulsewidth (float): The width of the pulse.

    Returns:
    np.ndarray: Lorentzian waveform.
    """
    # Calculate tau based on the pulse width
    tau = pulsewidth / 1.287
    # Compute the Lorentzian waveform
    lorentz = 1 / (1 + (t / tau) ** 2)
    return lorentz


def generate_sech(t, pulsewidth):
    """
    Generates a hyperbolic secant (sech) waveform.

    Parameters:
    t (array-like): Input array representing time values.
    pulsewidth (float): The width of the pulse.

    Returns:
    np.ndarray: Sech waveform.
    """
    # Calculate tau based on the pulse width
    tau = pulsewidth / 1.7627
    # Compute the sech waveform
    sechpulse = 1 / np.cosh(t / tau)
    return sechpulse
