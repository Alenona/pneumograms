import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq
from scipy.signal import hilbert


def spectrum_features(signal):
    N = len(signal)
    T = 0.02
    signal = (signal - np.mean(signal)) / np.std(signal)

    yf = fft(signal)
    xf = fftfreq(N, T)[:N // 2]
    amplitude_spectrum = 2.0 / N * np.abs(yf[0:N // 2])

    phase = np.angle(yf)
    phase_var = np.var(phase)

    idx = np.where((xf > 0.01) & (xf < 0.06))
    amplitudes = amplitude_spectrum[idx]
    frequencies = xf[idx]

    index = np.argmax(amplitudes)

    max_amplitude = amplitudes[index]
    dominant_freq = frequencies[index]

    rms_amplitude = np.sqrt(np.mean(np.abs(amplitudes) ** 2))  # Среднеквадратичное отклонение амплитуд

    total_energy = np.sum(np.abs(amplitudes) ** 2)  # Энергия спектра (сумма квадратов амплитуд)

    weighted_freq = np.sum(np.abs(amplitudes) * frequencies) / np.sum(
        np.abs(amplitudes))  # Средняя частота с весом по мощности

    nonzero_freqs = np.where(np.abs(amplitudes) > 0)[0]
    if len(nonzero_freqs) > 0:
        bandwidth = frequencies[nonzero_freqs[-1]] - frequencies[nonzero_freqs[0]]  # Ширина спектра
    else:
        bandwidth = 0

    harmonicity = max_amplitude / total_energy if total_energy != 0 else 0  # Коэффициент гармоничности (пиковая амплитуда к общей энергии)

    return amplitudes, [max_amplitude, dominant_freq, rms_amplitude, total_energy, weighted_freq, bandwidth, harmonicity, phase_var]


def hilbert_features(signal):
    hilb = hilbert(signal)
    envelope = np.abs(hilb)
    phase = np.angle(hilb)
    frequency = np.diff(np.unwrap(phase)) * (50 / (2.0 * np.pi))

    return np.mean(envelope), np.std(envelope), np.max(envelope), np.min(envelope), \
           np.max(envelope) / (np.min(envelope) + 1e-10), np.mean(frequency), np.std(frequency),\
           np.std(frequency) / (np.mean(frequency) + 1e-10)


df = pd.read_parquet('initial_data.parquet')
columns = ['amplitudes', 'fourier_features', 'hilbert_features']
features = pd.DataFrame(columns=columns)
features[['amplitudes', 'fourier_features']] = pd.DataFrame(df['signal'].apply(spectrum_features).to_list(), index=df.index)
features['hilbert_features'] = df['signal'].apply(hilbert_features)

features.to_parquet('FFT_Characteristics.parquet', index=False)
