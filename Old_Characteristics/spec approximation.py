import pandas as pd
import numpy as np
from scipy.fft import fft, fftfreq
from scipy.optimize import minimize
from scipy.stats import gennorm
from sklearn.cluster import KMeans


def data_transform(data):
    normalized_data = (data - np.mean(data)) / np.std(data)
    N = len(normalized_data)
    T = 0.02

    yf = fft(normalized_data) # выполняем быстрое преобразвание Фурье, чтобы извлечь амплитуды
    xf = fftfreq(N, T)[:N // 2] # ищем соответствующие частоты

    amplitudes = np.abs(yf[:N // 2])
    mask = xf <= 1
    xf_filtered = xf[mask]
    amplitudes_filtered = amplitudes[mask]

    to_det_distrib = []
    for i in range(len(amplitudes_filtered)):
        if xf_filtered[i]>0.1:
            current = [xf_filtered[i]] * round(amplitudes_filtered[i] / 100)
            to_det_distrib.extend(current)
    return np.array(to_det_distrib)


def log_like(params, data, p):
    [shape, loc, scale] = params
    likelihood = gennorm.pdf(data, shape, loc=loc, scale=scale)
    log_likelihood = np.sum(p*np.log(likelihood+1e-3))
    return -log_likelihood


def mle(data, params_initial, p):
    return minimize(log_like, params_initial, args=(data, p), method='Nelder-Mead', tol=1e-6)


def EM_Gennorm(data, max_iter):

    kmeans = KMeans(n_clusters=3, random_state=42).fit(data.reshape(-1, 1))
    labels = kmeans.labels_
    unique, counts = np.unique(labels, return_counts=True)

    sorted_clusters = unique[np.argsort(-counts)]

    new_labels = np.array([np.where(sorted_clusters == label)[0][0] for label in labels])

    [shape1, loc1, scale1] = gennorm.fit(data[new_labels==0])
    [shape2, loc2, scale2] = gennorm.fit(data[new_labels==1])

    pi = len(data[new_labels==0])/len(data)

    log_s_old = -np.inf
    for i in range(max_iter):

        # E-шаг:

        p_1 = gennorm.pdf(data, shape1, loc=loc1, scale=scale1)
        p_2 = gennorm.pdf(data, shape2, loc=loc2, scale=scale2)

        if np.sum(p_1)==0:
          p_1 = p_1+1e-2
        if np.sum(p_2)==0:
          p_2 = p_2+1e-2

        s = pi*p_1 +(1-pi)*p_2
        s = np.clip(s, 1e-6, None)
        log_s = np.sum(np.log(s))

        if abs(log_s-log_s_old)<1e-6:
          break
        log_s_old = log_s
        p_1 = pi*p_1 / s
        p_2 = (1-pi)*p_2 / s

        # M-шаг:

        [shape1, loc1, scale1] = mle(data, [shape1, loc1, scale1], p_1).x
        [shape2, loc2, scale2] = mle(data, [shape2, loc2, scale2], p_2).x

        pi = sum(p_1)/len(p_1)

    return shape1, loc1, scale1, shape2, loc2, scale2, pi


df = pd.read_parquet('initial_data.parquet')
features = pd.DataFrame(columns=['approx_params'])
features['approx_params'] = df['signal'].apply(lambda x: EM_Gennorm(data_transform(x), 50))
features.to_parquet('spec_approximation.parquet')

