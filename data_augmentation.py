import numpy as np
from scipy.signal import butter, lfilter
import librosa


def add_awgn(signal, snr_db=20):
    noise = np.random.normal(0, 1, len(signal))
    signal_power = np.mean(signal ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_power) * noise
    return signal + noise


def bandpass_filter(signal, sr=50, lowcut=0.1, highcut=15.0, order=5):
    nyquist = 0.5 * sr
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, signal)


def time_shift(signal, shift_max=0.1):
    shift = np.random.randint(-int(shift_max * len(signal)), int(shift_max * len(signal)))
    return np.roll(signal, shift)


def pitch_shift(signal, sr=50, pitch_range=0.1):
    steps = np.random.uniform(-pitch_range, pitch_range)
    return librosa.effects.pitch_shift(signal, sr=sr, n_steps=steps)