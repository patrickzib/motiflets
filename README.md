# Fast Pattern Discovery in Massive Time Series

This repository provides the reference implementation and experimental material 
for the paper **“Fast Pattern Discovery in Massive Time Series”**.

Time series motif discovery identifies repeated subsequences for
applications like ECG, EEG, and activity recognition. State-of-the-
art methods such as MASS are limited by quadratic time or memory
complexity, making large-scale analysis impractical. Current CPU
methods take nearly a day for 30 million points, and even GPU
acceleration requires hours. 
AMPED is a scalable anytime motif set finder using LSH-based pruning, anytime
processing, and a memory-efficient graph structure. On up to 25
large datasets (180k to 62M points, totaling 413 compute days across
all competitors), AMPED consistently ranks among the fastest and
most accurate methods. For million-scale data, it finds high-quality
motifs in under a minute with only 8 GB RAM, which is four orders
of magnitude faster than extrapolated MASS (21 days) and three
orders faster than MOMP (12 hours). Competitors either produced
low-quality results, needed excessive RAM, or crashed. AMPED
enables motif discovery in previously infeasible scenarios, including memory-constrained 
systems, and single-machine analytics,
while also reducing energy consumption, making it suitable for
sustainable large-scale data analysis.


## Repository Structure

This repository contains the full framework, benchmark datasets, and reproducible 
experiments used in the evaluation.


- `motiflets/`  
  Core implementation of the k-Motiflets algorithm.

- `notebooks/`  
  Jupyter notebooks demonstrating typical use cases and reproducing paper figures.

- `datasets/momp/`  
  Benchmark time series datasets used throughout the paper.

- `tests/csvs/`  
  Raw experimental results for all competing methods.

## AMPED (scalable Anytime Mining of Patterns under Euclidean Distance)

This paper introduces AMPED (scalable Anytime Mining of Pat-
terns under Euclidean Distance). It builds upon the Motiflets definition of 
motif sets but was systematically designed from the
ground up to exploit commodity multi-core hardware and SOTA
data structures while maintaining high precision. To overcome
the inherent quadratic-time bottleneck of motif search, AMPED
employs Locality-Sensitive Hashing (LSH) to aggressively prune

## Motiflets
Motif discovery aims to identify repeated patterns in time series data. A key 
difficulty in classical motif discovery is that both the motif length and the number 
of motif occurrences are unknown and must be inferred indirectly.
**k-Motiflets** introduce a new formulation of motif discovery that explicitly models 
the desired motif set size.
Intuitively, a *k-Motiflet* is the set of exactly `k` most similar subsequences of a 
given length. Instead of fixing a distance threshold and counting matches, k-Motiflets:

- take the motif size `k` as an explicit parameter, and
- maximize the internal similarity of the resulting motif set.

This turns classical motif discovery upside down. The parameter `k` has a clear and 
intuitive interpretation and is often known or easily estimated in real applications. 
This formulation enables fast algorithms with strong empirical performance on massive 
time series.

## Installation

Install the project directly from source.

### Build from Source

Clone the repository:

```bash
git clone https://github.com/patrickzib/motiflets.git
cd motiflets
````

Install the package:

```bash
pip install .
```

## Usage Example

```python
from motiflets.plotting import *

ml = Motiflets(
    ds_name,   # dataset name
    series,    # time series data
    n_jobs     # number of CPU cores
)

k_max = 20
motif_length = 100

dists, candidates, elbow_points = ml.fit_k_elbow(
    k_max,
    motif_length
)

ml.plot_motifset()
```



## Raw Experimental Results

All raw benchmark results reported in the paper are available in `tests/csvs/` for full 
reproducibility.
