import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import numpy as np
import pandas as pd

import numpy.fft as fft
from numba import objmode, prange, njit

import motiflets.motiflets as motif

np.set_printoptions(suppress=True)


def read_penguin_data():
    path = "../datasets/experiments/"
    series = pd.read_csv(path + "penguin.txt",
                         names=(["X-Acc", "Y-Acc", "Z-Acc",
                                 "4", "5", "6",
                                 "7", "Pressure", "9"]),
                         delimiter="\t", header=None)
    ds_name = "Penguins (Longer Snippet)"

    return ds_name, series


@njit(fastmath=True, cache=True, parallel=True)
def naive_sliding_dot_product(query, time_series):
    m = len(query)
    n = len(time_series)

    dot_product = np.zeros(n - m + 1, dtype=np.float64)
    for i in prange(dot_product.shape[0]):
        dot_product[i] = np.dot(query, time_series[i: i + m])

    return dot_product


@njit(fastmath=True, cache=True)
def sliding_dot_product_single_pad(query, time_series):
    m = len(query)
    n = len(time_series)

    time_series_add = 0
    if n % 2 == 1:
        time_series = np.concatenate((np.array([0]), time_series))
        time_series_add = 1

    q_add = 0
    if m % 2 == 1:
        query = np.concatenate((np.array([0]), query))
        q_add = 1

    query = query[::-1]
    query = np.concatenate((query, np.zeros(n - m + time_series_add - q_add)))
    trim = m - 1 + time_series_add
    with objmode(dot_product="float64[:]"):
        dot_product = fft.irfft(fft.rfft(time_series) * fft.rfft(query))
    return dot_product[trim:]



def test_sliding_dot_product_implementations():
    lengths = [
        #1_000,
        #1_001,
        #5_000,
        #5_001,
        10_000,
        10_001
    ]
    window_sizes = [100, 101]

    ds_name, B = read_penguin_data()
    times = np.zeros(3, dtype=np.float64)

    for i, length in enumerate(lengths):
        data = B.iloc[:length, 0].T.values
        print(f"Testing Dataset 'Penguin' of size {len(data)} {data.dtype}")
        for window_size in window_sizes:
            print("Current", length, "window size", window_size)

            n = length-window_size+1
            for i in np.arange(0, n):
                start = time.perf_counter()
                dot1 = (naive_sliding_dot_product(data[i:i+window_size], data))
                times[0] += time.perf_counter() - start
                #print(f"'Naive' Sliding Dot Product Time {time.perf_counter() - start:0.2f}")

                start = time.perf_counter()
                dot2 = (sliding_dot_product_single_pad(data[i:i+window_size], data))
                times[1] += time.perf_counter() - start
                #print(f"Sliding Dot Product 'Padding' Time {time.perf_counter() - start:0.2f}")

                start = time.perf_counter()
                dot3 = (motif._sliding_dot_product(data[i:i+window_size], data))
                times[2] += time.perf_counter() - start
                #print(
                #    f"Sliding Dot Product 'Motiflets' Time {time.perf_counter() - start:0.2f}")

                assert (dot1.shape == dot2.shape), f"'Padding is wrong"
                assert (np.allclose(dot1, dot2)), f"Padding is wrong {dot1, dot2}"

                assert (dot1.shape == dot3.shape), f"'Exponential Window' is wrong"
                assert (np.allclose(dot1, dot3)), f"'Exponential Window' is wrong"

    print("\n-----------------")
    print(f"'Naive' Sliding Dot Product Time {times[0]:0.1f}")
    print(f"Sliding Dot Product 'Window Pad' Time {times[1]:0.1f}")
    print(f"Sliding Dot Product 'Motiflets' Time {times[2]:0.1f}")
