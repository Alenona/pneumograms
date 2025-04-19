import pandas as pd
import os
import numpy as np


genders = pd.read_csv('пневмограммы/пневмограммы/genders.txt', delimiter='\t')
genders = genders.rename(columns={'№': 'number', 'Пол': 'gender'})
genders['gender'] = genders['gender'].replace({'м': 0, 'ж': 1})


columns = ["filename", "state", "gender", "signal"]
df = pd.DataFrame(columns=columns)


files = os.listdir('пневмограммы/пневмограммы/покой')
for file in files:
    if file.endswith(".txt"):
        num_str = os.path.splitext(file)[0][1:].replace('п', '')
        num = int(num_str)
        with open(os.path.join('пневмограммы/пневмограммы/покой', file), 'r') as f:
            data = f.readlines()
        data = np.array([float(line.strip().replace(',', '.')) for line in data])
        row = pd.DataFrame({
            'number': [num],
            'filename': [os.path.splitext(file)[0]],
            'state': [0],
            'gender': genders[genders.number == num].gender.values[0],
            'signal': [data]
        })
        df = pd.concat([df, row], ignore_index=True)


files = os.listdir('пневмограммы/пневмограммы/тревога')
for file in files:
    if file.endswith(".txt"):
        num_str = os.path.splitext(file)[0][1:].replace('т', '')
        num = int(num_str)
        with open(os.path.join('пневмограммы/пневмограммы/тревога', file), 'r') as f:
            data = f.readlines()
        data = np.array([float(line.strip().replace(',', '.')) for line in data])
        row = pd.DataFrame({
            'number': [num],
            'filename': [os.path.splitext(file)[0]],
            'state': [1],
            'gender': genders[genders.number == num].gender.values[0],
            'signal': [data]
        })
        df = pd.concat([df, row], ignore_index=True)

df.sort_values(by=['number', 'state'], inplace=True, ignore_index=True)

df.drop(columns=['number']).to_parquet('initial_data.parquet', index=False)
