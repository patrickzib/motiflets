import warnings

import numpy as np
from numba import njit

import motiflets.motiflets as ml

warnings.simplefilter("ignore")


@njit
def get_pair_motif(D_full):
    # ignore self
    for order in range(0, len(D_full)):
        D_full[order, order] = np.inf

    pair_motif_dist = np.min(D_full)
    pair_motif = np.argwhere(D_full == pair_motif_dist)

    return np.unique(pair_motif), pair_motif_dist, D_full


@njit
def filter_non_trivial_matches(motif_set, m):
    # filter trivial matches
    non_trivial_matches = []
    last_offset = - m
    for offset in np.sort(motif_set):
        if offset > last_offset + m / 2:
            non_trivial_matches.append(offset)
            last_offset = offset

    return np.array(non_trivial_matches)


def get_valmod_motif_set_ranged(
        data,
        file,
        motif_length,
        max_r=10, 
        steps=10
):
    D_full = ml.compute_distances_full(data, motif_length)
    m_half = motif_length // 2

    # get pair motif
    pair_motif, pair_motif_dist, D_full = get_pair_motif(D_full)
    yield pair_motif

    # perform range search around each offset
    last_size = 2
    for rr in np.arange(1, max_r + 1, max(1, int(max_r / steps))):
        motif_set = get_valmod_motif_set(data, file, motif_length, rr, D_full)

        # filter trivial matches
        if len(motif_set) > last_size:
            dist = ml.get_pairwise_extent(D_full, motif_set)
            yield motif_set

            for pos in motif_set:
                trivialMatchRange = (max(0, pos - m_half),
                             min(pos + m_half, len(D_full)))
                D_full[:, trivialMatchRange[0]:trivialMatchRange[1]] = np.inf

            last_size = len(motif_set)


def get_valmod_motif_set(
        data,
        file,
        motif_length,
        r,
        D_full=None):
    if D_full is None:
        D_full = ml.compute_distances_full(data, motif_length)

    # get pair motif
    pair_motif, pair_motif_dist, D_full = get_pair_motif(D_full)

    rr = r - pair_motif_dist
    motif_set = set()
    motif_set.update(pair_motif)
    for offset in pair_motif:
        result = np.argwhere(D_full[offset] <= rr).flatten()
        motif_set.update(result)

    return filter_non_trivial_matches(np.array(list(motif_set)), motif_length)
