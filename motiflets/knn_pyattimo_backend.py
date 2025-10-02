import os
from tabnanny import verbose

import psutil

import numpy as np


class PyAttimoNearestNeighbors:
    """
    PyAttimo-based nearest neighbor computations for motiflet discovery.

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

        self.pyattimo_delta = None
        if "pyattimo_delta" in kwargs:
            self.pyattimo_delta = kwargs["pyattimo_delta"]

        if "pyattimo_max_memory" in kwargs:
            self.pyattimo_max_memory = kwargs["pyattimo_max_memory"]
        else:
            self.pyattimo_max_memory = "8 GB"

    def compute_knns(self, X):
        """Compute k-nearest neighbors using PyAttimo motiflet discovery."""

        assert X.shape[0] == 1, \
            "PyAttimo can handle univariate data, only."

        try:
            import pyattimo
        except ImportError as e:
            raise PyAttimoError(f"Failed to import PyAttimo: {str(e)}")

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
            'max_memory': "8 GB"
        }

        if self.pyattimo_delta:
            attimo_args.update({
                'delta': self.pyattimo_delta,
                'stop_on_threshold': True,
                'fraction_threshold': np.log(n) / n,
            })

            if self.verbose:
                print(f"\tPyAttimo: Setting "
                      f"\n\t\tw={self.m}, "
                      f"\n\t\tdelta={self.pyattimo_delta}, "
                      f"\n\t\tsupport={attimo_args['support']}, "
                      f"\n\t\tmax_memory={attimo_args['max_memory']}, "
                      f"\n\t\texclusion_zone={attimo_args['exclusion_zone']}, "
                      f"\n\t\tstop_on_threshold={attimo_args['stop_on_threshold']}, "
                      f"\n\t\tfraction_threshold=log(n)/n")

        m_iter = pyattimo.MotifletsIterator(**attimo_args)

        try:
            for mot in m_iter:
                if verbose:
                    print(f"\t\t{mot}", flush=True)

                test_k = mot.support
                if test_k < self.k_max:
                    k_motiflet_distances[test_k] = mot.extent ** 2

                    # TODO: Use mot.lower_bound for confidence scores
                    k_motiflet_candidates[test_k] = np.array(mot.indices)

            if verbose:
                print(f"\t{len(k_motiflet_candidates[-1])}-Motiflet"
                      f"\n\t\tPos: {k_motiflet_candidates[-1]} "
                      f"\n\t\tExtent: {k_motiflet_distances[-1]}", flush=True)

            memory_usage = process.memory_info().rss / (1024 * 1024)  # MB

        except Exception as e:
            print(f"PyAttimo computation failed: {str(e)}", flush=True)

        if m_iter:
            del m_iter

        return k_motiflet_distances, k_motiflet_candidates, memory_usage
