import os
import numpy as np, pywt
from matplotlib import pyplot as plt


class DataGenerator:
    @staticmethod
    def generate_random_data(size: int) -> np.ndarray:
        return np.random.rand(size)


class FilterCalculator:
    @staticmethod
    def generate_h0() -> np.ndarray:
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
        y = np.zeros(len(data) * factor)
        y[::factor] = data
        return y

    @staticmethod
    def apply_difference_equation(data: np.ndarray, filter_coeffs: np.ndarray) -> np.ndarray:
        N = len(data)
        M = len(filter_coeffs)
        output = np.zeros(N)
        full_len = N + M - 1
        start_index_full = (full_len - N) // 2

        for n in range(N):
            n_full = n + start_index_full
            sum_val = 0.0
            for k in range(M):
                data_index = n_full - k
                if 0 <= data_index < N:
                    sum_val += filter_coeffs[k] * data[data_index]
            output[n] = sum_val
        return output


def plot_signal_pair(
        signal1: np.ndarray,
        signal2: np.ndarray,
         label1: str,
         label2: str,
         title1: str,
         title2: str,
         filename: str,
         marker1: str = '',
         marker2: str = ''
):
    fig, ax = plt.subplots(2, 1, figsize=(12, 10))

    # Plot for Signal 1
    ax[0].stem(signal1, label=label1)
    ax[0].set_title(title1)
    ax[0].set_xlabel('Sample')
    ax[0].set_ylabel('Amplitude')
    ax[0].legend()
    ax[0].grid(True)

    # Plot for Signal 2
    ax[1].stem(signal2, label=label2)
    ax[1].set_title(title2)
    ax[1].set_xlabel('Sample')
    ax[1].set_ylabel('Amplitude')
    ax[1].legend()
    ax[1].grid(True)

    # Save and close the figure
    fig.tight_layout()
    filepath = os.path.join('Results', filename)
    fig.savefig(filepath)
    plt.close(fig)
    print(f"Saved {filepath}")


def main():
    if not os.path.exists('Results'):
        os.mkdir('Results')

    # --- Data Generation ---
    data = DataGenerator.generate_random_data(size=1000)
    print("generated data:")
    print(data)
    print("-" * 30)

    # --- Filter Calculation ---
    h0 = FilterCalculator.generate_h0()
    h1 = FilterCalculator.calculate_h1(h0)
    g0 = FilterCalculator.calculate_g0(h0)
    g1 = FilterCalculator.calculate_g1(h1)

    print(f"h₀ (N={len(h0)}):")
    print(h0)
    print("-" * 30)
    print("h₁:")
    print(h1)
    print("-" * 30)
    print("g₀:")
    print(g0)
    print("-" * 30)
    print("g₁:")
    print(g1)
    print("-" * 30)

    # --- PyWT Comparison ---
    h1_pywt = np.array(pywt.Wavelet('db8').dec_hi)
    g0_pywt = np.array(pywt.Wavelet('db8').rec_lo)
    g1_pywt = np.array(pywt.Wavelet('db8').rec_hi)
    print(f"H₁ matches pywt.dec_hi?  {np.allclose(h1, h1_pywt)}")
    print(f"G₀ matches pywt.rec_lo?  {np.allclose(g0, g0_pywt)}")
    print(f"G₁ matches pywt.rec_hi?  {np.allclose(g1, g1_pywt)}")
    print("-" * 30)

    # --- Analysis Path ---
    theta0 = DataProcessor.apply_difference_equation(data, h0)
    theta1 = DataProcessor.apply_difference_equation(data, h1)

    v0 = DataProcessor.downsample(theta0, factor=2)
    v1 = DataProcessor.downsample(theta1, factor=2)

    # --- Synthesis Path ---
    f0 = DataProcessor.upsample(v0, 2)
    f1 = DataProcessor.upsample(v1, 2)

    y0 = DataProcessor.apply_difference_equation(f0, g0)
    y1 = DataProcessor.apply_difference_equation(f1, g1)

    x_hat = y0 + y1

    # --- Plotting ---
    plot_range = slice(100, 200)  # Define a zoom range

    # Call the reusable function for each pair
    plot_signal_pair(
        theta0[plot_range], theta1[plot_range],
        'theta0 (LPF output)',
        'theta1 (HPF output)',
        'Analysis Filter Output (theta0)',
        'Analysis Filter Output (theta1)',
        'theta_plots.png'
    )

    plot_signal_pair(
        v0[plot_range], v1[plot_range],
        'v0 (Approx Coeffs)',
        'v1 (Detail Coeffs)',
        'Downsampled Signal (v0)',
        'Downsampled Signal (v1)',
         'v_plots.png'
    )

    plot_signal_pair(
        f0[plot_range], f1[plot_range],
        'f0 (Upsampled v0)', 'f1 (Upsampled v1)',
         f'Upsampled Signal (f0) - Samples {plot_range.start}-{plot_range.stop}',
         f'Upsampled Signal (f1) - Samples {plot_range.start}-{plot_range.stop}',
         'f_plots.png',
            marker1='o', marker2='x'
    )

    plot_signal_pair(
        y0[plot_range], y1[plot_range],
 'y0 (Reconstructed LPF Path)', 'y1 (Reconstructed HPF Path)',
 'Reconstruction Filter Output (y0)', 'Reconstruction Filter Output (y1)',
       'y_plots.png'
    )

    plot_signal_pair(
        data[plot_range], x_hat[plot_range],
         'Original Data (x)', 'Reconstructed Data (x_hat)',
         f'Original Signal (Samples {plot_range.start}-{plot_range.stop})',
            f'Reconstructed Signal (Samples {plot_range.start}-{plot_range.stop})',
          'reconstruction_plots.png',
          marker1='.', marker2=''
    )

    # --- Error Calculation ---
    mse = np.mean((data - x_hat) ** 2)
    print(f"\nReconstruction Mean Squared Error (MSE): {mse:.2e}")


if __name__ == "__main__":
    main()