# -*- coding: utf-8 -*-
"""
LatentMotif motif discovery algorithm wrapper.
Origin: https://github.com/grrvlr/TSMD/
"""

import numpy as np

import warnings

warnings.filterwarnings('ignore')

from numba import njit

class LatentMotif(object):
    """LatentMotif algorithm for motif discovery.

    Parameters
    ----------
    n_patterns : int 
        Number of patterns to detect.
    radius : float
        Threshold factor for pattern inclusion.
    wlen : int
        Window length.
    alpha : float, optional (default=1.0)
        Regularization parameter.
    learning_rate : float, optional (default=0.1)
        Learning rate. 
    n_iterations :int, optional (default=100): 
        Number of gradient iteration.
    n_starts : int, optional (default=10): 
        Number of trials. 
    verbose : bool, optional (default=False) : 
        Verbose. 
    Attributes
    ----------
    prediction_mask_ : np.ndarray of shape (n_patterns, n_samples)
        Binary mask indicating the presence of motifs across the signal.  
        Each row corresponds to one discovered motif, and each column to a time step.  
        A value of 1 means the motif is present at that time step, and 0 means it is not.
    """

    def __init__(self, n_patterns: int, wlen: int, radius: float, alpha=1.0,
                 learning_rate=0.1, n_iterations=100, n_starts=1,
                 verbose=False) -> None:

        self.n_patterns = n_patterns
        self.wlen = wlen
        self.radius = radius
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.n_starts = n_starts
        self.verbose = verbose

    def _freq(self, patterns: np.ndarray) -> float:  # verified
        """Compute the frequency score of the given patterns.

        Parameters
        ----------
        patterns : np.ndarray 
            Array of shape (n_patterns, wlen) representing the patterns.

        Returns
        -------
        freq : float
            Frequency score. Measures the similarity of the given patterns to the internal set.
        """
        dist = np.sum((self.set_[:, np.newaxis, :] - patterns[np.newaxis, ...]) ** 2,
                      axis=2)
        exp_dist = np.exp(-self.alpha / self.radius * dist)
        freq = 1 / (self.n_patterns * self.set_size_) * np.sum(exp_dist)
        return freq

    def _pen(self, patterns):  # verified
        """Compute a penalty score between patterns.

        Parameters
        ----------
        patterns : np.ndarray 
            Array of shape (n_patterns, wlen) representing the patterns.

        Returns
        -------
        pen : float
            Penalty score. 
        """
        if self.n_patterns > 1:
            dist = np.sum((patterns[:, np.newaxis, :] - patterns[np.newaxis, ...]) ** 2,
                          axis=2)
            pen_m = np.where(dist < 2 * self.radius,
                             (1 - dist / (2 * self.radius)) ** 2, 0)
            pen = 2 / (self.n_patterns * (self.n_patterns - 1)) * np.sum(
                np.triu(pen_m, k=1))
        else:
            pen = 0
        return pen

    def _score(self, patterns):  # verified
        """Compute the overall score for the given patterns.
        The score is defined as the frequency minus the penalty.

        Parameters
        ----------
        patterns : np.ndarray 
            Array of shape (n_patterns, wlen) representing the patterns.

        Returns
        -------
        score: float 
            Overall score
        """
        return self._freq(patterns) - self._pen(patterns)

    def fit(self, signal: np.ndarray) -> None:
        """Fit LatentMotif
        
        Parameters
        ----------
        signal : numpy array of shape (n_samples, )
            The input samples (time series length).
        
        Returns
        -------
        self : object
            Fitted estimator.
        """
        # initialization
        self.signal_ = signal
        self.set_ = np.lib.stride_tricks.sliding_window_view(signal, self.wlen)
        self.set_ = (self.set_ - np.mean(self.set_, axis=1).reshape(-1, 1)) / np.std(
            self.set_, axis=1).reshape(-1, 1)
        self.set_size_ = self.set_.shape[0]
        self.score_ = -np.inf
        self.patterns_ = np.zeros((self.n_patterns, self.wlen))

        if self.verbose:
            print("Start Trials")
        for i in range(self.n_starts):
            patterns = one_fit_(
                self.n_patterns, self.wlen, self.n_iterations, self.set_, self.set_size_, self.alpha, self.radius,
                self.learning_rate
            )
            score = self._score(patterns)
            if self.verbose:
                print(f"Trial: {i + 1}/{self.n_starts}, score : {score}")
            if score > self.score_:
                self.score_ = score
                self.patterns_ = patterns

        if self.verbose:
            print(f"Successfully finished, best score: {self.score_}")

        return self

    @property
    def prediction_mask_(self) -> np.ndarray:
        dist = np.sum(
            (self.set_[:, np.newaxis, :] - self.patterns_[np.newaxis, ...]) ** 2,
            axis=2)
        idx_lsts = []
        for line in dist.T:
            idxs = np.arange(line.shape[0])
            idx_lst = []
            t_distance = np.min(line)
            while t_distance < self.radius:
                # try:
                # local next neighbor
                t_idx = np.argmin(line)
                t_distance = line[t_idx]
                if line[t_idx] < self.radius:
                    idx_lst.append(idxs[t_idx])
                    # remove window
                    remove_idx = np.arange(max(0, t_idx - self.wlen + 1),
                                           min(len(line), t_idx + self.wlen))
                    line[remove_idx] = np.inf

                # except:
                # break
            idx_lsts.append(idx_lst)

        mask = np.zeros((self.n_patterns, self.signal_.shape[0]))
        for i, p_idx in enumerate(idx_lsts):
            for idx in p_idx:
                mask[i, idx:idx + self.wlen] = 1

                # remove null lines
        mask = mask[~np.all(mask == 0, axis=1)]

        return mask, idx_lsts

@njit(cache=True)
def one_fit_(n_patterns, wlen, n_iterations, set_, set_size_, alpha, radius, learning_rate):
    patterns = np.random.randn(n_patterns, wlen)
    rate_adapt = np.zeros((n_patterns, wlen))

    for i in range(n_iterations):
        if n_patterns > 1:
            div = _freq_derivative_numba(
                set_, n_patterns, set_size_, alpha, radius,
                patterns) - _pen_derivative_numba(radius, n_patterns, patterns)

        else:
            div = _freq_derivative_numba(
                set_, n_patterns, set_size_, alpha, radius, patterns)

        rate_adapt += div ** 2
        patterns -= learning_rate / np.sqrt(rate_adapt) * div

    return patterns


@njit(cache=True)
def _pen_derivative_numba(radius, n_patterns, patterns):
    """Compute the derivative of the penalty score with respect to the patterns.

    Parameters
    ----------
    patterns : np.ndarray
        Array of shape (n_patterns, wlen) representing the patterns.

    Returns
    -------
    div_pen : float
        Penalty score derivative
    """
    diff = patterns[:, np.newaxis, :] - patterns[np.newaxis, ...]
    dist = np.sum(diff ** 2, axis=2)
    pen_m = np.where(dist < 2 * radius, 2 * radius - dist, 0)
    div_pen = -2 / (radius ** 2 * n_patterns * (
                n_patterns - 1)) * np.sum(pen_m[..., np.newaxis] * diff, axis=0)
    return div_pen


@njit(cache=True)
def _freq_derivative_numba(set_, n_patterns, set_size_, alpha, radius, patterns):
    """Compute the derivative of the frequency score with respect to the patterns.

    Parameters
    ----------
    patterns : np.ndarray
        Array of shape (n_patterns, wlen) representing the patterns.

    Returns
    -------
    div_freq : float
        Frequency score derivative
    """
    diff = set_[:, np.newaxis, :] - patterns[np.newaxis, ...]
    exp_dist = np.exp(-alpha / radius * np.sum(diff ** 2, axis=2))
    div_freq = -2 * alpha / (n_patterns * set_size_ * radius) * np.sum(exp_dist[..., np.newaxis] * diff, axis=0)
    return div_freq
