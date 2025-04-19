import pandas as pd
import numpy as np
import librosa


def mfcc_mean(signal):

  mfccs = librosa.feature.mfcc(y=signal, sr=0.02, n_mfcc=12)
  mfcc_mean = np.mean(mfccs, axis=1)
  return mfcc_mean


df = pd.read_parquet('initial_data.parquet')
mfcc_features = pd.DataFrame(columns=['mfcc_mean'])
mfcc_features['mfcc_mean'] = df['signal'].apply(mfcc_mean)

mfcc_features.to_parquet('MFCC.parquet')
