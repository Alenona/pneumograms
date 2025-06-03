import pywt
import numpy as np
import pandas as pd


def CWT(signal):
    scales = np.linspace(125, 250)
    coeffs, freqs = pywt.cwt(signal, scales, 'cmor1.5-1.0', sampling_period=0.02)
    return coeffs, freqs


def frequency_varience(coeffs):
    f = [np.fft.fft(coeff) for coeff in coeffs]
    dl = [sum(abs(f[i]-f[i-1])) for i in range(1, 50)]
    return dl


def time_varience(coeffs):
    coeffs_transposed = coeffs.T
    d2 = [abs(coeff).argmax() for coeff in coeffs_transposed]
    spec = abs(np.fft.fft(d2))[:25]
    return spec


df = pd.read_parquet('initial_data.parquet')

features = pd.DataFrame(columns=['CWT_coeffs', 'CWT_freqs', 'CWT_freq_var', 'CWT_time_var'])
features[['CWT_coeffs', 'CWT_freqs']] = pd.DataFrame(df['signal'].apply(CWT).to_list(), index=df.index)
features['CWT_freq_var'] = features['CWT_coeffs'].apply(frequency_varience)
features['CWT_time_var'] = features['CWT_coeffs'].apply(time_varience)

features[['CWT_freq_var', 'CWT_time_var']].to_parquet('CWT_Characteristics.parquet')