import os
import psutil

import numpy as np

def compute_knns_pyattimo(
        X,
        m,
        k_max,
        slack=0.5,
        **kwargs
):
    assert X.shape[0] == 1, \
        "PyAttimo can handle univariate data, only."

    import pyattimo

    n = X.shape[-1] - m + 1

    pid = os.getpid()
    process = psutil.Process(pid)

    k_motiflet_distances = np.zeros(k_max)
    k_motiflet_candidates = np.empty(k_max, dtype=object)
    memory_usage = 0
    exclusion_m = int(m * slack)

    # Prepare common arguments
    attimo_args = {
        'ts': X.flatten(),
        'w': m,
        'support': k_max - 1,
        'exclusion_zone': exclusion_m,
        'max_memory': "8 GB"
    }

    if "pyattimo_delta" in kwargs:
        pyattimo_delta = kwargs["pyattimo_delta"]
        attimo_args.update({
            'delta': pyattimo_delta,
            'stop_on_threshold': True,
            'fraction_threshold': np.log(n) / n,
        })
        print(f"\tPyAttimo: Setting "
              f"\n\t\tw={m}, "
              f"\n\t\tdelta={pyattimo_delta}, "
              f"\n\t\tsupport={attimo_args['support']}, "
              f"\n\t\tmax_memory={attimo_args['max_memory']}, "
              f"\n\t\texclusion_zone={attimo_args['exclusion_zone']}, "
              f"\n\t\tstop_on_threshold={attimo_args['stop_on_threshold']}, "
              f"\n\t\tfraction_threshold=log(n)/n")

    m_iter = pyattimo.MotifletsIterator(**attimo_args)

    try:
        for mot in m_iter:
            print(f"\t\t{mot}", flush=True)
            test_k = mot.support
            if test_k < k_max:
                k_motiflet_distances[test_k] = mot.extent ** 2

                # TODO: Use mot.lower_bound for confidence scores
                k_motiflet_candidates[test_k] = np.array(mot.indices)

        print(f"\t{len(k_motiflet_candidates[-1])}-Motiflet"
              f"\n\t\tPos: {k_motiflet_candidates[-1]} "
              f"\n\t\tExtent: {k_motiflet_distances[-1]}", flush=True)
        memory_usage = process.memory_info().rss / (1024 * 1024)  # MB

    except:
        print("Caught exception in pyattimo", flush=True)

    del m_iter

    return k_motiflet_distances, k_motiflet_candidates, memory_usage
