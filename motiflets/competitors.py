# -*- coding: utf-8 -*-
"""Implementation of competitor methods.
"""

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
        motif_set = get_valmod_motif_set(data, motif_length, rr, D_full)

        # filter trivial matches
        if len(motif_set) > last_size:
            # _ = ml.get_pairwise_extent(D_full, motif_set)
            yield motif_set

            for pos in motif_set:
                trivialMatchRange = (max(0, pos - m_half),
                                     min(pos + m_half, len(D_full)))
                D_full[:, trivialMatchRange[0]:trivialMatchRange[1]] = np.inf

            last_size = len(motif_set)


def get_valmod_motif_set(
        data,
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


def get_k_motifs_ranged(
        data,
        motif_length,
        max_r=10):
    D_full = ml.compute_distances_full(data, motif_length)

    # get pair motif
    pair_motif, pair_motif_dist, _ = get_pair_motif(D_full)
    yield (pair_motif)

    last_size = 2

    for r in range(1, max_r + 1, max(1, int(max_r / 10))):
        k_motifset = get_k_motifs(data, motif_length, r, D_full)

        if len(k_motifset) > last_size:
            pairwise_dist = ml.get_pairwise_extent(D_full, k_motifset)
            yield (k_motifset)

        last_size = max(last_size, len(k_motifset))


def get_k_motifs(
        data,
        motif_length,
        r,
        D_full=None):
    if D_full is None:
        D_full = ml.compute_distances_full(data, motif_length)

    # allow subsequence itself
    np.fill_diagonal(D_full, 0)

    cardinality = -1
    k_motif_dist_var = -1
    k_motifset = []
    for order, dist in enumerate(D_full):
        motif_set = np.argwhere(dist <= r).flatten()
        if len(motif_set) > cardinality:
            # filter trivial matches
            motif_set = filter_non_trivial_matches(motif_set, motif_length)

            # Break ties by variance of distances
            dist_var = np.var(dist[motif_set])
            if len(motif_set) > cardinality or \
                    (dist_var < k_motif_dist_var and len(motif_set) == cardinality):
                cardinality = len(motif_set)
                k_motifset = motif_set
                k_motif_dist_var = dist_var

    return k_motifset
