import os
os.environ['NUMBA_CACHE_DIR'] = '/tmp/motifs'

import sys

sys.path.insert(0, "../")
sys.path.insert(0, "../../")

import numpy as np
import utils as ut


deltas = [0.1]
k_max = 10

def main():
    lengths = np.array([properties[-1] for properties in list(ut.filenames.values())])
    sorted_idx = np.argsort(lengths)
    for filename in np.array(list(ut.filenames.keys()))[sorted_idx]:
        ds_name, length, _, _ = ut.filenames[filename]
        print(f"Running: {filename, ds_name}")
        data = ut.read_mat(filename)

        # pyattimo
        backend = "pyattimo"
        for delta in deltas:
            ut.run_safe(
                ds_name=filename,
                series=data,
                l_range=[length],
                k_max=k_max,
                backend=backend,
                pyattimo_delta=delta,
                pyattimo_max_memory="20 GB"
            )


if __name__ == "__main__":
    main()
