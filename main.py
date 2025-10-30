import os

import numpy as np
from matplotlib import pyplot as plt
import pywt


class DataGenerator:
    @staticmethod
    def generate_random_data(size: int) -> np.ndarray:
        return np.random.rand(size)


class FilterCalculator:
    @staticmethod
    def generate_h0() -> np.ndarray:
        """
        Generates the H₀ filter coefficients from the 'db8' wavelet.
        For db8, N=16 (which is even).
        """
        return np.array(pywt.Wavelet('db8').dec_lo)

    @staticmethod
    def calculate_h1(h0: np.ndarray) -> np.ndarray:
        n_indices = np.arange(len(h0))
        alternating_sign = (-1) ** n_indices
        return -1 * alternating_sign * np.flip(h0)

    @staticmethod
    def calculate_g0(h0: np.ndarray) -> np.ndarray:
        return np.flip(h0)

    @staticmethod
    def calculate_g1(h1: np.ndarray) -> np.ndarray:
        return np.flip(h1)

class DataProcessor:
    @staticmethod
    def downsample(data: np.ndarray, factor: int) -> np.ndarray:
        return data[::factor]

    @staticmethod
    def upsample(data: np.ndarray, factor: int) -> np.ndarray:
        """
        Upsamples a 1D array by inserting 'factor-1' zeros between samples.
        """
        y = np.zeros(len(data) * factor)
        y[::factor] = data
        return y

    @staticmethod
    def filter_data(data: np.ndarray, filter_coeffs: np.ndarray, mode: str = 'full') -> np.ndarray:
        return np.convolve(data, filter_coeffs, mode=mode)

def main():
    if not os.path.exists('Results'):
        os.mkdir('Results')

    data = DataGenerator.generate_random_data(size=1000)
    print("generated data:")
    print(data)
    print("-" * 30)

    h0 = FilterCalculator.generate_h0()
    print(f"h₀ (N={len(h0)}):")
    print(h0)
    print("-" * 30)

    # 2. Calculate the other filters
    h1 = FilterCalculator.calculate_h1(h0)
    g0 = FilterCalculator.calculate_g0(h0)
    g1 = FilterCalculator.calculate_g1(h1)

    # 3. Print the results
    print("h₁:")
    print(h1)
    print("-" * 30)

    print("g₀:")
    print(g0)
    print("-" * 30)

    print("g₁:")
    print(g1)
    print("-" * 30)

    # 1. Get the standard orthogonal filters from pywt
    h1_pywt = np.array(pywt.Wavelet('db8').dec_hi)
    g0_pywt = np.array(pywt.Wavelet('db8').rec_lo)
    g1_pywt = np.array(pywt.Wavelet('db8').rec_hi)

    # 2. Perform comparisons
    h1_matches = np.allclose(h1, h1_pywt)
    g0_matches = np.allclose(g0, g0_pywt)
    g1_matches = np.allclose(g1, g1_pywt)

    print(f"H₁ matches pywt.dec_hi?  {h1_matches}")
    print(f"G₀ matches pywt.rec_lo?  {g0_matches}")
    print(f"G₁ matches pywt.rec_hi?  {g1_matches}")
    print("-" * 30)

    theta0 = DataProcessor.filter_data(data, h0, mode='same')
    theta1 = DataProcessor.filter_data(data, h1, mode='same')

    v0 = DataProcessor.downsample(theta0, factor=2)
    v1 = DataProcessor.downsample(theta1, factor=2)

    f0 = DataProcessor.upsample(v0, 2)
    f1 = DataProcessor.upsample(v1, 2)

    y0 = DataProcessor.filter_data(f0, g0, mode='same')
    y1 = DataProcessor.filter_data(f1, g1, mode='same')

    x_hat = y0 + y1

    mse = np.mean((data - x_hat) ** 2)
    print(f"\nReconstruction Mean Squared Error (MSE): {mse:.2e}")


if __name__ == "__main__":
    main()
