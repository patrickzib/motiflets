import os
os.environ['NUMBA_CACHE_DIR'] = '/tmp/motifs'

import sys

sys.path.insert(0, "../")
sys.path.insert(0, "../../")

import numpy as np
import utils as ut


# ks = [5, 10, 20]
deltas = [0.1]

# Compare:
# https://github.com/erikbern/ann-benchmarks/blob/main/ann_benchmarks/algorithms/faiss_hnsw/config.yml
faiss_efConstruction = [500]
faiss_efSearch = [400]  # 600, 800
faiss_M = [64]   # 32, 64, 96

# Compare:
# https://github.com/erikbern/ann-benchmarks/blob/main/ann_benchmarks/algorithms/faiss/config.yml
# faiss_nlist = [32]  # using sqrt(n) by default
faiss_nprobe = [10]  # [10, 50, 100, 200]

# Compare:
# https://github.com/erikbern/ann-benchmarks/blob/main/ann_benchmarks/algorithms/faiss/config.yml
# faiss_nlist = [32]  # using sqrt(n) by default
faiss_nprobe = [10]  # [10, 50, 100, 200]

# Compare:
# https://github.com/erikbern/ann-benchmarks/blob/main/ann_benchmarks/algorithms/pynndescent/config.yml
pynndescent_n_neighbors = [60]
pynndescent_leaf_size = [48]
pynndescent_pruning_degree_multiplier = [2.0]
pynndescent_diversify_prob = [0.0]
pynndescent_n_search_trees = [1]
pynndescent_search_epsilon = [0.2]  # [0.0, 0.04, 0.08, 0.12, 0.16, 0.2, 0.24, 0.28, 0.32, 0.36]

# LSH
faiss_nbits = [4]

# Annoy
annoy_n_trees = [100]
annoy_search_k = [-1]


k_max = 10

def main():
    lengths = np.array([properties[-1] for properties in list(ut.filenames.values())])
    sorted_idx = np.argsort(lengths)
    for filename in np.array(list(ut.filenames.keys()))[sorted_idx]:
        ds_name, length, _, _ = ut.filenames[filename]
        print(f"Running: {filename, ds_name}")
        data = ut.read_mat(filename)

        # # pyattimo
        # backend = "pyattimo"
        # for delta in deltas:
        #     for subsampling in [16, 32, 64]:
        #         ut.run_safe(
        #                filename, data, l_range, k_max, backend,
        #                pyattimo_delta=delta, subsampling=subsampling
        #         )

        # Running FAISS
        backend = "faiss"

        # faiss_index = "LSH"
        # for nbits in faiss_nbits:
        #     print(f"\n\tRunning faiss {faiss_index} {nbits}")
        #     ut.run_safe(
        #         filename,
        #         data,
        #         l_range,
        #         k_max,
        #         backend,
        #         faiss_index=faiss_index,
        #         faiss_nbits=nbits,
        #      )

        faiss_index = "HNSW"
        for M in faiss_M:
            for efConstruction in faiss_efConstruction:
                for efSearch in faiss_efSearch:
                    print(f"\n\tRunning faiss {faiss_index} {M} {efConstruction} {efSearch}.", flush=True)
                    ut.ut.run_safe(
                        filename,
                        data,
                        l_range,
                        k_max,
                        backend,
                        faiss_index=faiss_index,
                        faiss_M=M,
                        faiss_efConstruction=efConstruction,
                        faiss_efSearch=efSearch
                        )

        # faiss_index = "IVF"
        # for nprobe in faiss_nprobe:
        #     print(f"\n\tRunning faiss {faiss_index} {nprobe}")
        #     ut.run_safe(
        #         filename,
        #         data,
        #         l_range,
        #         k_max,
        #         backend,
        #         faiss_index=faiss_index,
        #         faiss_nprobe=nprobe
        #      )

        # faiss_index = "IVFPQ"
        # for nprobe in faiss_nprobe:
        #     print(f"\n\tRunning faiss {faiss_index} {nprobe}")
        #     ut.run_safe(
        #         filename,
        #         data,
        #         l_range,
        #         k_max,
        #         backend,
        #         faiss_index=faiss_index,
        #         faiss_nprobe=nprobe
        #     )


        # # Running pynndescent
        # backend = "pynndescent"
        # for n_neighbors in pynndescent_n_neighbors:
        #     for leaf_size in pynndescent_leaf_size:
        #         for pruning_degree_multiplier in pynndescent_pruning_degree_multiplier:
        #             for diversify_prob in pynndescent_diversify_prob:
        #                 for n_search_trees in pynndescent_n_search_trees:
        #                     for search_epsilon in pynndescent_search_epsilon:
        #                         print(f"\n\tRunning pynndescent")
        #                         ut.run_safe(
        #                             filename,
        #                             data,
        #                             l_range,
        #                             k_max,
        #                             backend,
        #                             pynndescent_n_neighbors=n_neighbors,
        #                             pynndescent_leaf_size=leaf_size,
        #                             pynndescent_pruning_degree_multiplier=pruning_degree_multiplier,
        #                             pynndescent_diversify_prob=diversify_prob,
        #                             pynndescent_n_search_trees=n_search_trees,
        #                             pynndescent_search_epsilon=search_epsilon
        #                         )

        # # Running ANNOY
        # backend = "annoy"
        #
        # for n_trees in annoy_n_trees:
        #     for search_k in annoy_search_k:
        #         print(f"\n\tRunning annoy {n_trees} {search_k}")
        #         ut.run_safe(
        #             filename,
        #             data,
        #             l_range,
        #             k_max,
        #             backend,
        #             annoy_n_trees=n_trees,
        #             annoy_search_k=search_k
        #          )

        # scalable
        # backend = "scalable"
        # ut.run_safe(
        #   filename, data, l_range, k_max, backend, subsampling=10
        # )

        # # subsampling
        # backend = "scalable"
        # for subsampling in [8, 16]:
        #    ut.run_safe(
        #       filename, data,
        #       l_range=l_range, k_max=k_max,
        #       backend=backend,
        #       subsampling=subsampling
        #    )


if __name__ == "__main__":
    main()
