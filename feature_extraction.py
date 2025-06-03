import librosa
import numpy as np


def extract_mfcc(signal, sr=50, n_mfcc=40):
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)
    return mfcc


def extract_chroma_stft(signal, sr=50):
    stft = np.abs(librosa.stft(signal, n_fft=64, hop_length=16))
    chroma = librosa.feature.chroma_stft(S=stft, sr=sr, tuning=0)
    return chroma


def extract_mel_spectrogram(signal, sr=50, n_mels=40):
    mel_spec = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=n_mels)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db


def extract_spectral_rolloff(signal, sr=50, roll_percent=0.85):
    S = np.abs(librosa.stft(signal, n_fft=1024))
    rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.85)
    return rolloff[0]


def extract_zero_crossing_rate(signal):
    zcr = librosa.feature.zero_crossing_rate(signal-np.mean(signal), frame_length=100, hop_length=50)
    return zcr[0]


def extract_features(signal, sr=50):
    mfcc = extract_mfcc(signal)
    mel_db = extract_mel_spectrogram(signal)
    chroma = extract_chroma_stft(signal)
    rolloff = extract_spectral_rolloff(signal)
    zcr = extract_zero_crossing_rate(signal)

    x_xgb = np.hstack([
        np.mean(mfcc, axis=1),
        np.mean(mel_db, axis=1),
        np.mean(chroma, axis=1),
        np.std(mfcc, axis=1),
        np.std(mel_db, axis=1),
        np.std(chroma, axis=1),
        [np.mean(rolloff), np.std(rolloff)],
        [np.mean(zcr), np.std(zcr)],
    ])

    return {
        'mfcc': mfcc,
        'mel': mel_db,
        'chroma': chroma,
        'rolloff': rolloff,
        'zcr': zcr,
        'x_xgb': x_xgb
    }