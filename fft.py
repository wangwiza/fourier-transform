import numpy as np
import matplotlib.pyplot as plt

# The naive DFT algorithm
def naive_dft(x):
    N = len(x)
    X = np.zeros(N, dtype=np.complex_)
    # both k and n go from 0 to N-1
    for k in range(N):
        for n in range(N):
            X[k] += x[n] * np.exp(-2j * np.pi * k * n / N)
    return X

# The Cooley-Tukey FFT algorithm
def fft(x):
    N = len(x)
    if N <= 1:
        return x
    X_even = fft(x[::2])
    X_odd = fft(x[1::2])
    factor = np.exp(-2j * np.pi * np.arange(N) / N)
    return np.concatenate([X_even + factor[:N // 2] * X_odd,
                           X_even + factor[N // 2:] * X_odd])

# TODO: Inverse FFT
