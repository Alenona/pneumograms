import pandas as pd


initial_data = pd.read_parquet('initial_data.parquet')
FFT = pd.read_parquet('FFT_Characteristics.parquet')
CWT = pd.read_parquet('CWT_Characteristics.parquet')
MFCC = pd.read_parquet('MFCC.parquet')
spec_approx = pd.read_parquet('spec_approximation.parquet')

total_features = pd.concat([initial_data.drop(columns=['signal']), FFT, CWT, MFCC, spec_approx], axis=1)

total_features.to_parquet('total_features.parquet')