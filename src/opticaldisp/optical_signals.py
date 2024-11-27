import numpy as np


class OpticalSignal:
    def __init__(self, wavelength, ts, et):
        """
        Initialize the OpticalSignal class.

        Parameters:
        wavelength (float): Wavelength of the optical signal in nanometers.
        ts (float): Sampling period in seconds.
        Et (array-like): Complex amplitude envelope of the signal in time.
        """
        self.wavelength = wavelength
        self.ts = ts
        self.et = np.array(et, dtype=complex)

    @property
    def pt(self):
        """Power of the signal in time."""
        return np.abs(self.et) ** 2

    @property
    def meanpower(self):
        """Mean power of the signal."""
        return np.mean(np.abs(self.et) ** 2)

    @property
    def n(self):
        """Number of samples in each polarization."""
        return self.et.size

    @property
    def frequency(self):
        """Signal frequency based on the wavelength."""
        c = 299792485  # Speed of light in m/s
        return c / self.wavelength
