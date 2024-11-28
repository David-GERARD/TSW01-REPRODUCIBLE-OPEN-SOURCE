import numpy as np
from scipy.constants import c  # Speed of light


def apply_dispersion(sigin, cd):
    """
    Applies chromatic dispersion to an input optical signal.

    Parameters:
    SigIn (OpticalSignal): Input signal object with attributes `Et`, `wavelength`, and `ts`.
    CD (float): Dispersion coefficient in ps/nm.

    Returns:
    OpticalSignal: Output signal with dispersion applied.
    """
    # Copy the input signal to the output
    sigOut = sigin
    N = sigin.et.size

    # Conversion from D to beta_2
    beta2L = -(sigin.wavelength**2) / (2 * np.pi * c) * (cd * 1e-12 / 1e-9)

    # Frequency vector in the Fourier domain
    omega = (
        2
        * np.pi
        / (N * sigin.ts)
        * np.concatenate((np.arange(0, N // 2), np.arange(-N // 2, 0)))
    )

    # Dispersion operator in the frequency domain
    dd = 1j / 2 * beta2L * omega**2

    # Apply dispersion in the frequency domain
    sigOut.Et = np.fft.ifft(sigin.et)
    sigOut.Et = np.exp(dd) * sigOut.et
    sigOut.Et = np.fft.fft(sigOut.et)

    return sigOut
