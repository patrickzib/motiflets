# import os
# import numpy as np
# import psutil
#
# class ExactNearestNeighbors:
#     """
#     Exact nearest neighbor computations for motiflet discovery based on MASS.
#
#     Parameters:
#     -----------
#
#     """
#
#     def __init__(
#             self,
#             m, k,
#             n_jobs=n_jobs,
#             slack=slack,
#             backend="scalable",
#             distance=distance,
#             distance_single=distance_single,
#             distance_preprocessing=distance_preprocessing):
#
#         self.m = m
#         self.k = k
#         self.n_jobs = n_jobs
#         self.slack = slack
#         self.distance = distance
#         self.distance_single = distance_single
#         self.distance_preprocessing = distance_preprocessing
#
#         self.backend = backend
#
#
#     def compute_knns(self, X):
#         """Compute k-nearest neighbors using exact implementations."""
#         pid = os.getpid()
#         process = psutil.Process(pid)
#         memory_usage = 0.0
#
#         if backend == "scalable":
#             # uses pairwise comparisons to compute the distances
#             call_to_distances = compute_distances_with_knns
#         elif backend == "sparse":
#             # uses sparse backend with sparse matrix
#             call_to_distances = compute_distances_with_knns_sparse
#         else:
#             # computes the full matrix
#             call_to_distances = compute_distances_with_knns_full
#
#         D_full, knns = call_to_distances(
#             X, self.m, self.k,
#             n_jobs=self.n_jobs,
#             slack=self.slack,
#             distance=self.distance,
#             distance_single=self.distance_single,
#             distance_preprocessing=self.distance_preprocessing
#         )
#
#         memory_usage = process.memory_info().rss / (1024 * 1024)  # MB
#         return D_full, knns, memory_usage
#
#
#
#
# @njit(nogil=True, fastmath=True, cache=True, parallel=True)
# def compute_distances_with_knns(
#         time_series,
#         m,
#         k,
#         exclude_trivial_match=True,
#         n_jobs=4,
#         slack=0.5,
#         distance=znormed_euclidean_distance,
#         distance_single=znormed_euclidean_distance_single,
#         distance_preprocessing=sliding_mean_std
# ):
#     """ Compute the full Distance Matrix between all pairs of subsequences of a
#         multivariate time series.
#
#         Computes pairwise distances between n-m+1 subsequences, of length, extracted
#         from the time series, of length n.
#
#         Z-normed ED is used for distances.
#
#         This implementation is in O(n^2) by using the sliding dot-product.
#
#         Parameters
#         ----------
#         time_series : array-like
#             The time series
#         m : int
#             The window length
#         k : int
#             Number of nearest neighbors
#         exclude_trivial_match : bool (default: True)
#             Trivial matches will be excluded if this parameter is set
#         n_jobs : int (default: 4)
#             Number of jobs to be used.
#         slack: float (default: 0.5)
#             Defines an exclusion zone around each subsequence to avoid trivial matches.
#             Defined as percentage of m. E.g. 0.5 is equal to half the window length.
#         distance: callable (default: znormed_euclidean_distance)
#                 The distance function to be computed.
#         distance_preprocessing: callable (default: sliding_mean_std)
#                 The distance preprocessing function to be computed.
#
#         Returns
#         -------
#         D : 2d array-like
#             The O(n^2) z-normed ED distances between all pairs of subsequences
#         knns : 2d array-like
#             The k-nns for each subsequence
#
#     """
#     # Input dim must be 2d
#     assert time_series.ndim == 2, "Dimensionality is not correct"
#
#     dims = time_series.shape[0]
#     n = np.int32(time_series.shape[-1] - m + 1)
#     n_jobs = max(1, min(n // 8, n_jobs))  # Cannot use more jobs than length of the ts
#
#     halve_m = 0
#     if exclude_trivial_match:
#         halve_m = np.int32(m * slack)
#
#     D_knn = np.zeros((n, k), dtype=np.float64)
#     knns = np.full((n, k), -1, dtype=np.int32)
#
#     bin_size = np.int32(np.ceil(time_series.shape[-1] / n_jobs))
#
#     preprocessing = []
#     dot_first = []
#
#     for d in np.arange(dims):
#         ts = time_series[d]
#         preprocessing.append(distance_preprocessing(ts, m))
#         dot_first.append(_sliding_dot_product(ts[:m], ts))
#
#     # first pass, computing the k-nns
#     for idx in prange(n_jobs):
#         dot_rolled = np.zeros((dims, n), dtype=np.float64)
#         dot_prev = np.zeros((dims, n), dtype=np.float64)
#
#         start = np.int32(idx * bin_size)
#         end = np.int32(min(start + bin_size, n))
#
#         for order in np.arange(start, end, ):
#             dist = np.zeros(n, dtype=np.float64)
#             for d in np.arange(dims):
#                 ts = time_series[d, :]
#                 if order == start:
#                     # O(n log n) operation
#                     dot_rolled[d] = _sliding_dot_product(ts[start:start + m], ts)
#                 else:
#                     # constant time O(1) operations
#                     dot_rolled[d] = np.roll(dot_prev[d], 1) \
#                                     + ts[order + m - 1] * ts[m - 1:n + m] \
#                                     - ts[order - 1] * np.roll(ts[:n], 1)
#                     dot_rolled[d][0] = dot_first[d][order]
#
#                 dists = distance(dot_rolled[d], n, m, preprocessing[d], order, halve_m)
#                 for i in range(len(dists)):
#                     dist[i] += dists[i]
#                 dot_prev[d] = dot_rolled[d]
#
#             knn = _argknn(dist, k, m, slack=slack)
#             D_knn[order, :len(knn)] = dist[knn]
#             knns[order, :len(knn)] = knn
#
#     return D_knn, knns
#
#
#
#
# @njit(fastmath=True, cache=True, nogil=True)
# def compute_distances_with_knns_sparse(
#         time_series,
#         m,
#         k,
#         exclude_trivial_match=True,
#         n_jobs=4,
#         slack=0.5,
#         distance=znormed_euclidean_distance,
#         distance_single=znormed_euclidean_distance_single,
#         distance_preprocessing=sliding_mean_std
# ):
#     """ Compute the full Distance Matrix between all pairs of subsequences of a
#         multivariate time series.
#
#         Computes pairwise distances between n-m+1 subsequences, of length, extracted
#         from the time series, of length n.
#
#         Z-normed ED is used for distances.
#
#         This implementation is in O(n^2) by using the sliding dot-product.
#
#         Parameters
#         ----------
#         time_series : array-like
#             The time series
#         m : int
#             The window length
#         k : int
#             Number of nearest neighbors
#         exclude_trivial_match : bool (default: True)
#             Trivial matches will be excluded if this parameter is set
#         n_jobs : int (default: 4)
#             Number of jobs to be used.
#         slack: float (default: 0.5)
#             Defines an exclusion zone around each subsequence to avoid trivial matches.
#             Defined as percentage of m. E.g. 0.5 is equal to half the window length.
#         distance: callable (default: znormed_euclidean_distance)
#                 The distance function to be computed.
#         distance_preprocessing: callable (default: sliding_mean_std)
#                 The distance preprocessing function to be computed.
#
#         Returns
#         -------
#         D : 2d array-like
#             The O(n^2) z-normed ED distances between all pairs of subsequences
#         knns : 2d array-like
#             The k-nns for each subsequence
#
#     """
#     # Input dim must be 2d
#     assert time_series.ndim == 2, "Dimensionality is not correct"
#
#     dims = time_series.shape[0]
#     n = np.int32(time_series.shape[-1] - m + 1)
#     n_jobs = max(1, min(n // 8, n_jobs))  # Cannot use more jobs than length of the ts
#
#     halve_m = 0
#     if exclude_trivial_match:
#         halve_m = int(m * slack)
#
#     bin_size = np.int32(np.ceil(time_series.shape[-1] / n_jobs))
#
#     D_knn, knns = compute_distances_with_knns(
#         time_series,
#         m,
#         k,
#         exclude_trivial_match,
#         n_jobs,
#         slack,
#         distance,
#         distance_preprocessing
#     )
#
#     preprocessing = []
#     dot_first = []
#     for dim in np.arange(time_series.shape[0]):
#         preprocessing.append(distance_preprocessing(time_series[dim], m))
#         dot_first.append(_sliding_dot_product(time_series[dim, :m], time_series[dim]))
#
#     # TODO no sparse matrix support in numba. Thus we use this hack
#     D_bool = [Dict.empty(key_type=types.int32, value_type=types.uint16) for _ in
#               range(n)]
#
#     # Store an upper bound for each k-nn distance
#     kth_extent = compute_upper_bound(
#         time_series, D_knn, knns, k, m,
#         distance_single, preprocessing,
#     )
#
#     # Parallelizm does not work, as Dict is not thread safe :/
#     for order in np.arange(0, n):
#         # memorize which pairs are needed
#         for ks, dist in zip(knns[order], D_knn[order]):
#             D_bool[order][ks] = True
#
#             bound = False
#             k_index = -1
#             for kk in range(len(kth_extent) - 1, 0, -1):
#                 if D_knn[order, kk] <= kth_extent[kk]:
#                     bound = True
#                     k_index = kk + 1
#                     break
#             if bound:
#                 for ks2 in knns[order, :k_index]:
#                     D_bool[ks][ks2] = True
#
#     D_sparse = List()
#     for i in range(n):
#         D_sparse.append(Dict.empty(key_type=types.int32, value_type=types.float32))
#
#     # second pass, filling only the pairs needed
#     for idx in prange(n_jobs):
#         dot_rolled = np.zeros((dims, n), dtype=np.float32)
#         dot_prev = np.zeros((dims, n), dtype=np.float32)
#
#         start = idx * bin_size
#         end = min(start + bin_size, n)
#
#         for order in np.arange(start, end, dtype=np.int32):
#             dist = np.zeros(n, dtype=np.float32)
#             for d in np.arange(dims):
#                 ts = time_series[d, :]
#                 if order == start:
#                     # O(n log n) operation
#                     dot_rolled[d] = (_sliding_dot_product(ts[start:start + m], ts))
#                 else:
#                     # constant time O(1) operations
#                     dot_rolled[d] = np.roll(dot_prev[d], 1) \
#                                     + ts[order + m - 1] * ts[m - 1:n + m] \
#                                     - ts[order - 1] * np.roll(ts[:n], 1)
#                     dot_rolled[d][0] = dot_first[d][order]
#
#                 dist += distance(dot_rolled[d], n, m, preprocessing[d], order, halve_m)
#                 dot_prev[d] = dot_rolled[d]
#
#             # fill the k-nns now with the distances computed
#             for key in D_bool[order]:
#                 D_sparse[order][key] = dist[key]
#
#     return D_sparse, knns
#
#
#
#
# @njit(nogil=True, fastmath=True, cache=True, parallel=True)
# def compute_distances_with_knns_full(
#         time_series,
#         m,
#         k,
#         exclude_trivial_match=True,
#         n_jobs=4,
#         slack=0.5,
#         distance=znormed_euclidean_distance,
#         distance_single=znormed_euclidean_distance_single,
#         distance_preprocessing=sliding_mean_std
# ):
#     """Compute the full Distance Matrix between all pairs of subsequences.
#
#         Computes pairwise distances between n-m+1 subsequences, of length, extracted from
#         the time series, of length n.
#
#         Z-normed ED is used for distances.
#
#         This implementation is in O(n^2) by using the sliding dot-product.
#
#         Parameters
#         ----------
#         time_series : array-like
#             The time series
#         m : int
#             The window length
#         k : int
#             Number of nearest neighbors
#         exclude_trivial_match : bool (default: True)
#             Trivial matches will be excluded if this parameter is set
#         n_jobs : int (default: 4)
#             Number of jobs to be used.
#         slack: float (default: 0.5)
#             Defines an exclusion zone around each subsequence to avoid trivial matches.
#             Defined as percentage of m. E.g. 0.5 is equal to half the window length.
#         distance: callable (default: znormed_euclidean_distance)
#                 The distance function to be computed.
#         distance_preprocessing: callable (default: sliding_mean_std)
#                 The distance preprocessing function to be computed.
#
#         Returns
#         -------
#         D : 2d array-like
#             The O(n^2) z-normed ED distances between all pairs of subsequences
#         knns : 2d array-like
#             The k-nns for each subsequence
#
#     """
#     # Input dim must be 2d
#     assert time_series.ndim == 2, "Dimensionality is not correct"
#
#     dims = time_series.shape[0]
#     n = np.int32(time_series.shape[-1] - m + 1)
#     n_jobs = max(1, min(n // 8, n_jobs))  # Cannot use more jobs than length of the ts
#
#     halve_m = 0
#     if exclude_trivial_match:
#         halve_m = np.int32(m * slack)
#
#     D = np.zeros((n, n), dtype=np.float64)
#     knns = np.full((n, k), -1, dtype=np.int32)
#
#     bin_size = np.int32(np.ceil(time_series.shape[-1] / n_jobs))
#
#     for idx in prange(n_jobs):
#         start = idx * bin_size
#         end = min(start + bin_size, n)
#
#         for d in np.arange(dims):
#             ts = time_series[d]
#             preprocessing = distance_preprocessing(ts, m)
#             dot_first = _sliding_dot_product(ts[:m], ts)
#
#             dot_prev = None
#             for order in np.arange(start, end):
#                 if order == start:
#                     # O(n log n) operation
#                     dot_rolled = _sliding_dot_product(ts[start:start + m], ts)
#                 else:
#                     # constant time O(1) operations
#                     dot_rolled = np.roll(dot_prev, 1) \
#                                  + ts[order + m - 1] * ts[m - 1:n + m] \
#                                  - ts[order - 1] * np.roll(ts[:n], 1)
#                     dot_rolled[0] = dot_first[order]
#
#                 dist = distance(dot_rolled, n, m, preprocessing, order, halve_m)
#                 D[order] += dist
#                 dot_prev = dot_rolled
#
#         for order in np.arange(start, end):
#             knn = _argknn(D[order], k, m, slack=slack)
#             knns[order, :len(knn)] = knn
#
#     return D, knns
#
#
#
# @njit(fastmath=True, cache=True)
# def compute_upper_bound(
#         ts, D_knn, knns, k, m,
#         distance_single, preprocessing,
# ):
#     kth_extent = np.zeros(k, dtype=np.float64)
#     kth_extent[0] = np.inf
#
#     for kk in range(1, len(kth_extent)):
#         # kk is the kk-th NN
#         # The motiflet candidate has thus kk+1 elements (including the query itself)
#         best_knn_pos = np.argmin(D_knn[:, kk])
#         candidate = knns[best_knn_pos, :kk + 1]
#         kth_extent[kk] = get_pairwise_extent_raw(
#             ts, candidate, m,
#             distance_single,
#             preprocessing)
#
#         # extent must be within the diameter of the sphere
#         kth_nn_min = np.min(D_knn[:, kk])
#         if kth_extent[kk] > 4 * kth_nn_min or kth_extent[kk] < kth_nn_min:
#             kth_extent[kk] = kth_nn_min
#
#         # assert kth_extent[kk] <= 4 * kth_nn_min
#         # assert kth_extent[kk] >= kth_nn_min
#
#     return kth_extent
