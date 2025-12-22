import os
import psutil
import numpy as np

#from motiflets.distances import *
#from motiflets.motiflets import _sliding_dot_product, _argknn, get_pairwise_extent_raw
#from numba import njit, prange


class SCAMPINearestNeighbors:
    """
    SCAMPI-based nearest neighbor computations for motiflet discovery.

    Parameters:
    -----------
    motif_length : int
        Length of motifs to discover (must be positive)
    k_max : int
        Maximum number of motiflets to discover (must be positive)
    slack : float, default=0.5
        Exclusion zone factor for motif discovery (0.0 to 1.0)
    verbose : bool, default=True
        Whether to print computation progress
    """

    def __init__(
            self,
            m,
            k_max,
            slack=0.5,
            verbose=True,
            **kwargs):

        self.m = m
        self.k_max = k_max
        self.slack = slack
        self.verbose = verbose

        self.scampi_delta = None
        if "scampi_delta" in kwargs:
            self.scampi_delta = kwargs["scampi_delta"]

        if "scampi_max_memory" in kwargs:
            self.scampi_max_memory = kwargs["scampi_max_memory"]
            print(f"Setting SCAMPI max memory to {self.scampi_max_memory}")
        else:
            self.scampi_max_memory = "8 GB"


    def compute_knns(self, X):
        """Compute k-nearest neighbors using SCAMPI motiflet discovery."""

        assert X.shape[0] == 1, \
            "SCAMPI can handle univariate data, only."

        try:
            import pyattimo
        except ImportError as e:
            raise PyAttimoError(f"Failed to import SCAMPI: {str(e)}")

        n = X.shape[-1] - self.m + 1

        pid = os.getpid()
        process = psutil.Process(pid)

        k_motiflet_distances = np.zeros(self.k_max, dtype=np.float64)
        k_motiflet_candidates = np.empty(self.k_max, dtype=object)
        memory_usage = 0.0

        # Prepare common arguments
        attimo_args = {
            'ts': X.flatten(),
            'w': self.m,
            'support': self.k_max - 1,
            'exclusion_zone': int(self.m * self.slack),
            'max_memory': self.scampi_max_memory,
            # 'observability_file': "observe.csv"
        }

        if self.scampi_delta:
            attimo_args.update({
               'delta': self.scampi_delta,
               'stop_on_threshold': True,
               'fraction_threshold': np.log(n) / n
            })

            #attimo_args.update({
            #    'delta': self.scampi_delta,
            #    'stop_on_threshold': False,
            # })

            if self.verbose:
                print(f"\tSCAMPI: Setting "
                      f"\n\t\tw={self.m}, "
                      f"\n\t\tdelta={self.scampi_delta}, "
                      f"\n\t\tsupport={attimo_args['support']}, "
                      f"\n\t\tmax_memory={attimo_args['max_memory']}, "
                      f"\n\t\texclusion_zone={attimo_args['exclusion_zone']}, "
                      f"\n\t\tstop_on_threshold={attimo_args['stop_on_threshold']}, "
                      # f"\n\t\tfraction_threshold=log(n)/n", flush=True
                , flush=True)

        m_iter = pyattimo.MotifletsIterator(**attimo_args)

        try:
            print("\tComputing motiflets with SCAMPI...", flush=True)
            for mot in m_iter:
                if self.verbose:
                    print(f"\t\t{mot}", flush=True)

                test_k = mot.support
                if test_k < self.k_max:
                    k_motiflet_distances[test_k] = mot.extent ** 2

                    # TODO: Use mot.lower_bound for confidence scores
                    k_motiflet_candidates[test_k] = np.array(mot.indices)

            if self.verbose:
                print(f"\t{len(k_motiflet_candidates[-1])}-Motiflet"
                      f"\n\t\tPos: {k_motiflet_candidates[-1]} "
                      f"\n\t\tExtent: {k_motiflet_distances[-1]}", flush=True)

            memory_usage = process.memory_info().rss / (1024 * 1024)  # MB

        except Exception as e:
            print(f"SCAMPI computation failed: {str(e)}", flush=True)

        if m_iter:
            del m_iter

        return k_motiflet_distances, k_motiflet_candidates, memory_usage


# @njit(cache=True, parallel=True)
# def compute_knn(
#         ts,
#         motiflets,
#         m,
#         k,
#         slack=0.5,
#         distance=znormed_euclidean_distance,
#         distance_single=znormed_euclidean_distance_single,
#         distance_preprocessing=sliding_mean_std,
# ):
#     halve_m = np.int32(m * slack)
#     n = ts.shape[-1] - m + 1
#
#     preprocessing = distance_preprocessing(ts, m)
#
#     knns = np.zeros((len(motiflets), k), dtype=np.int32)
#     extents = np.zeros(len(motiflets), dtype=np.float64)
#
#     for i in prange(len(motiflets)):
#         start = motiflets[i]
#         if start < len(ts) - m + 1:
#             dot_rolled = _sliding_dot_product(
#                 ts[start:start + m],
#                 ts,
#             )
#             dist = distance(dot_rolled, n, m, preprocessing, start, halve_m)
#             knns[i] = _argknn(dist, k, m, slack=slack)
#
#             extents[i] = get_pairwise_extent_raw_1d(
#                 ts, knns[i], m, distance_single, preprocessing)
#         else:
#             extents[i] = np.inf
#
#     min_pos = np.argmin(extents)
#     best_motiflet = knns[min_pos]
#     min_extent = extents[min_pos]
#
#     return best_motiflet, min_extent
#
#
# @njit(cache=True)
# def get_pairwise_extent_raw_1d(
#         series, motifset_pos, motif_length,
#         distance_single, preprocessing):
#     """Computes the extent of the motifset via pairwise comparisons.
#
#     Parameters
#     ----------
#     series : array-like
#         The time series
#     motifset_pos : array-like
#         The motif set start-offsets
#     motif_length : int
#         The motif length
#     upperbound : float, default: np.inf
#         Upper bound on the distances. If passed, will apply admissible pruning
#         on distance computations, and only return the actual extent, if it is lower
#         than `upperbound`
#
#     Returns
#     -------
#     motifset_extent : float
#         The extent of the motif set, if smaller than `upperbound`, else np.inf
#     """
#
#     if -1 in motifset_pos:
#         return np.inf
#
#     motifset_extent = np.float64(0.0)
#
#     for ii in np.arange(len(motifset_pos) - 1):
#         i = motifset_pos[ii]
#         a = series[i:i + motif_length]
#
#         for jj in np.arange(ii + 1, len(motifset_pos)):
#             j = motifset_pos[jj]
#             b = series[j:j + motif_length]
#             dist = distance_single(a, b, i, j, preprocessing)
#             motifset_extent = max(motifset_extent, dist)
#
#     return motifset_extent