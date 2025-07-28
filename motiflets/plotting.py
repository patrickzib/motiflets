# -*- coding: utf-8 -*-
"""Plotting utilities.
"""

__author__ = ["patrickzib"]

import os
import time

import matplotlib
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
from scipy.stats import zscore
from tsdownsample import MinMaxLTTBDownsampler

import motiflets.motiflets as ml
from motiflets.distances import *

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


class Motiflets:

    def __init__(
            self,
            ds_name,
            series,
            ground_truth=None,
            elbow_deviation=1.00,
            distance="znormed_ed",
            slack=0.5,
            n_jobs=-1,
            backend="scalable"
    ):
        """Computes the AU_EF plot to extract the best motif lengths

            This is the method to find and plot the characteristic motif-lengths, for k in
            [2...k_max], using the area AU-EF plot.

            Details are given within the paper 5.2 Learning Motif Length l.

            Parameters
            ----------
            ds_name: String
                Name of the time series for displaying
            series: array-like
                the TS
            ground_truth: pd.Series
                Ground-truth information as pd.Series.
            elbow_deviation : float, default=1.00
                The minimal absolute deviation needed to detect an elbow.
                It measures the absolute change in deviation from k to k+1.
                1.05 corresponds to 5% increase in deviation.
            distance: str (default="znormed_ed")
                The name of the distance function to be computed.
                Available options are:
                    - 'znormed_ed' or 'znormed_euclidean' for z-normalized ED
                    - 'ed' or 'euclidean' for the "normal" ED.
            slack: float
                Defines an exclusion zone around each subsequence to avoid trivial matches.
                Defined as percentage of m. E.g. 0.5 is equal to half the window length.
            n_jobs : int
                Number of jobs to be used.
            backend : String, default="scalable"
                The backend to use. As of now 'scalable', 'sparse' and 'default' are supported.
                Use 'default' for the original exact implementation with excessive memory,
                Use 'scalable' for a scalable, exact implementation with less memory,
                Use 'sparse' for a scalable, exact implementation with more memory.

            Returns
            -------
            best_motif_length: int
                The motif length that maximizes the AU-EF.
        """
        self.ds_name = ds_name
        self.series = series
        self.elbow_deviation = elbow_deviation
        self.slack = slack
        self.ground_truth = ground_truth

        n_jobs = os.cpu_count() if n_jobs < 1 else n_jobs
        self.n_jobs = n_jobs

        # distance function used
        self.distance_preprocessing, self.distance, self.distance_single \
            = map_distances(distance)
        self.backend = backend

        self.motif_length_range = None
        self.motif_length = 0
        self.all_extrema = []
        self.all_elbows = []
        self.all_top_motiflets = []
        self.all_dists = []

        self.motif_length = 0
        self.memory_usage = 0
        self.k_max = 0
        self.dists = []
        self.motiflets = []
        self.elbow_points = []

    def fit_motif_length(
            self,
            k_max,
            motif_length_range,
            subsample=2,
            # plot=True,
            # plot_elbows=False,
            # plot_motifs_as_grid=True,
            # plot_best_only=True
    ):
        """Computes the AU_EF plot to extract the best motif lengths

            This is the method to find and plot the characteristic motif-lengths, for k in
            [2...k_max], using the area AU-EF plot.

            Details are given within the paper 5.2 Learning Motif Length l.

            Parameters
            ----------
            k_max: int
                use [2...k_max] to compute the elbow plot.
            motif_length_range: array-like
                the interval of lengths
            subsample: int (default=2)
                the subsample factor

            Returns
            -------
            best_motif_length: int
                The motif length that maximizes the AU-EF.

            """

        self.motif_length_range = motif_length_range
        self.k_max = k_max

        self.motif_length = plot_motif_length_selection(
            k_max,
            self.series,
            motif_length_range,
            self.ds_name,
            elbow_deviation=self.elbow_deviation,
            slack=self.slack,
            subsample=subsample,
            n_jobs=self.n_jobs,
            distance=self.distance,
            distance_single=self.distance_single,
            distance_preprocessing=self.distance_preprocessing,
            backend=self.backend

        )

        return self.motif_length

    def fit_k_elbow(
            self,
            k_max,
            motif_length=None,  # if None, use best_motif_length
            filter=True,
            plot_elbows=True,
            plot_motifs_as_grid=True,
    ):
        """Plots the elbow-plot for k-Motiflets.

            This is the method to find and plot the characteristic k-Motiflets within range
            [2...k_max] for given a `motif_length` using elbow-plots.

            Details are given within the paper Section 5.1 Learning meaningful k.

            Parameters
            ----------
            k_max: int
                use [2...k_max] to compute the elbow plot (user parameter).
            motif_length: int
                the length of the motif (user parameter)
            filter: bool, default=True
                filters overlapping motiflets from the result,
            plot_elbows: bool, default=False
                plots the elbow ploints into the plot

            Returns
            -------
            Tuple
                dists:          distances for each k in [2...k_max]
                candidates:     motifset-candidates for each k
                elbow_points:   elbow-points

            """
        self.k_max = k_max

        if motif_length is None:
            motif_length = self.motif_length
        else:
            self.motif_length = motif_length

        self.dists, self.motiflets, self.elbow_points, self.memory_usage = plot_elbow(
            k_max,
            self.series,
            ds_name=self.ds_name,
            motif_length=motif_length,
            plot_elbows=plot_elbows,
            plot_grid=plot_motifs_as_grid,
            ground_truth=self.ground_truth,
            filter=filter,
            n_jobs=self.n_jobs,
            elbow_deviation=self.elbow_deviation,
            slack=self.slack,
            distance=self.distance,
            distance_single=self.distance_single,
            distance_preprocessing=self.distance_preprocessing,
            backend=self.backend,
        )

        return self.dists, self.motiflets, self.elbow_points

    def plot_dataset(self, path=None):
        fig, ax = plot_dataset(
            self.ds_name,
            self.series,
            show=path is None,
            ground_truth=self.ground_truth)

        if path is not None:
            plt.savefig(path)
            plt.show()

        return fig, ax

    def plot_motifset(self, path=None, elbow_point=None):

        if self.dists is None or self.motiflets is None or self.elbow_points is None:
            raise Exception("Please call fit_k_elbow first.")

        if elbow_point is None:
            elbow_point = self.elbow_points[-1]

        fig, ax = plot_motifset(
            self.ds_name,
            self.series,
            motifsets=self.motiflets[elbow_point],
            dist=self.dists[elbow_point],
            motif_length=self.motif_length,
            show=path is None)

        if path is not None:
            plt.savefig(path)
            plt.show()

        return fig, ax


def convert_to_2d(
        series
):
    if series.ndim == 1:
        # print('Warning: The input dimension must be 2d.')
        if isinstance(series, pd.Series):
            series = series.to_frame().T
        elif isinstance(series, (np.ndarray, np.generic)):
            series = np.arange(series.shape[-1])
    if series.shape[0] > series.shape[1]:
        raise ('Warning: The input shape is wrong. Dimensions should be on rows. '
               'Try transposing the input.')

    return series


def as_series(data, index_range, index_name):
    """Coverts a time series to a series with an index.

    Parameters
    ----------
    data : array-like
        The time series raw data as numpy array
    index_range :
        The index to use
    index_name :
        The name of the index to use (e.g. time)

    Returns
    -------
    series : PD.Series

    """
    series = pd.Series(data=data, index=index_range)
    series.index.name = index_name
    return series


def plot_dataset(
        ds_name,
        data,
        ground_truth=None,
        show=True
):
    """Plots a time series.

    Parameters
    ----------
    ds_name: String
        The name of the time series
    data: array-like
        The time series
    ground_truth: pd.Series
        Ground-truth information as pd.Series.
    show: boolean
        Outputs the plot

    """
    return plot_motifset(ds_name, data, ground_truth=ground_truth, show=show)


def append_all_motif_sets(df, motif_sets, method_name, D_full):
    """Utility function.

    Parameters
    ----------
    df: pd.DataFrame
        a dataframe to append to
    motif_sets: 2d-array-like
        The motif-sets to append under row `method_name`
    method_name: String
        The column to append as
    D_full:
        The distance matrix

    Returns
    -------
    df: pd.DataFrame
        the dataframe with appended data

    """

    filtered_motif_sets = [m for m in motif_sets if m is not None]
    extent = [ml.get_pairwise_extent(D_full, motiflet) for motiflet in
              filtered_motif_sets]
    count = [len(motiflet) for motiflet in filtered_motif_sets]

    for m, e, c in zip(filtered_motif_sets, extent, count):
        entry = {"Method": method_name, "Motif": m, "Extent": e, "k": c}
        df = df.append(entry, ignore_index=True)
    return df


def plot_motifset(
        ds_name,
        data,
        motifsets=None,
        dist=None,
        motif_length=None,
        ground_truth=None,
        show=True):
    """Plots the data and the found motif sets.

    Parameters
    ----------
    ds_name: String,
        The name of the time series
    data: array-like
        The time series data
    motifsets: array like
        One found motif set
    dist: array like
        The distances (extents) for each motif set
    motif_length: int
        The length of the motif
    ground_truth: pd.Series
        Ground-truth information as pd.Series.
    show: boolean
        Outputs the plot

    """
    # set_sns_style(font_size)
    # sns.set(font_scale=3)
    sns.set(font="Calibri")
    sns.set_style("white")

    # turn into 2d array
    data = convert_to_2d(data)

    if motifsets is not None:
        git_ratio = [4]
        for _ in range(len(motifsets)):
            git_ratio.append(1)

        fig, axes = plt.subplots(2, 1 + len(motifsets),
                                 sharey="row",
                                 sharex=False,
                                 figsize=(
                                     10 + 2 * len(motifsets),
                                     5 + (data.shape[0] + len(motifsets)) // 2),
                                 squeeze=False,
                                 gridspec_kw={
                                     'width_ratios': git_ratio,
                                     'height_ratios': [10, 3]})  # 5 for rolling stone?
    elif ground_truth is not None:
        fig, axes = plt.subplots(2, 1,
                                 sharey="row",
                                 sharex=False,
                                 figsize=(20, 5 + data.shape[0] // 2),
                                 squeeze=False,
                                 gridspec_kw={
                                     'width_ratios': [4],
                                     'height_ratios': [10, 1]})
    else:
        fig, axes = plt.subplots(1, 1, squeeze=False,
                                 figsize=(20, 5 + data.shape[0] // 2))

    if ground_truth is None:
        ground_truth = []

    data_index, data_raw = ml.pd_series_to_numpy(data)
    data_raw_sampled, data_index_sampled = data_raw, data_index

    factor = 1
    if data_raw.shape[-1] > 2_000:
        data_raw_sampled = np.zeros((data_raw.shape[0], 2_000))
        for i in range(data_raw.shape[0]):
            index = MinMaxLTTBDownsampler().downsample(
                np.ascontiguousarray(data_raw[i]), n_out=2_000)
            data_raw_sampled[i] = data_raw[i, index]

        data_index_sampled = data_index[index]
        factor = max(1, data_raw.shape[-1] / data_raw_sampled.shape[-1])
        if motifsets is not None:
            motifsets_sampled = list(map(lambda x: np.int32(x // factor), motifsets))
    else:
        motifsets_sampled = motifsets

    color_offset = 1
    offset = 0
    tick_offsets = []
    axes[0, 0].set_title(ds_name, fontsize=22)

    for dim in range(data_raw.shape[0]):
        dim_raw = zscore(data_raw[dim])
        dim_raw_sampled = zscore(data_raw_sampled[dim])
        offset -= 1.2 * (np.max(dim_raw_sampled) - np.min(dim_raw_sampled))
        tick_offsets.append(offset)

        _ = sns.lineplot(
            x=data_index_sampled,
            y=dim_raw_sampled + offset,
            ax=axes[0, 0],
            linewidth=0.5,
            color="gray",
            errorbar=("ci", None),
            estimator=None
        )
        sns.despine()

        if motifsets is not None:
            for i, motifset in enumerate(motifsets_sampled):
                if motifset is not None:
                    motif_length_sampled = np.int32(max(2, motif_length // factor))
                    for a, pos in enumerate(motifset):
                        _ = sns.lineplot(
                            ax=axes[0, 0],
                            x=data_index_sampled[
                              pos: pos + motif_length_sampled],
                            y=dim_raw_sampled[
                              pos: pos + motif_length_sampled] + offset,
                            linewidth=3,
                            color=sns.color_palette("tab10")[
                                (color_offset + i) % len(
                                    sns.color_palette("tab10"))],
                            errorbar=("ci", None),
                            estimator=None)

                        motif_length_disp = motif_length

                        axes[0, 1 + i].set_title(
                            ("Motif Set " + str(i + 1)) + "\n" +
                            "k=" + str(len(motifset)) +
                            ", l=" + str(motif_length_disp),
                            fontsize=18)

                        df = pd.DataFrame()
                        df["time"] = range(0, motif_length_disp, 1)

                        for aa, pos in enumerate(motifsets[i]):
                            values = np.zeros(len(df["time"]), dtype=np.float32)
                            value = dim_raw[pos:pos + motif_length_disp:1]
                            values[:len(value)] = value

                            df[str(aa)] = (values - values.mean()) / (
                                    values.std() + 1e-4) + offset

                        df_melt = pd.melt(df, id_vars="time")
                        _ = sns.lineplot(
                            ax=axes[0, 1 + i],
                            data=df_melt,
                            errorbar=("ci", 99),
                            n_boot=1,
                            lw=1,
                            color=sns.color_palette("tab10")[
                                (color_offset + i) % len(sns.color_palette("tab10"))],
                            x="time",
                            y="value")

        gt_count = 0
        y_labels = []
        motif_set_count = 0 if motifsets is None else len(motifsets)

        for aaa, column in enumerate(ground_truth):
            for offsets in ground_truth[column]:
                for off in offsets:
                    ratio = 0.8
                    start = np.int32(off[0] // factor)
                    end = np.int32(off[1] // factor)
                    if end - 1 < dim_raw_sampled.shape[0]:
                        rect = Rectangle(
                            (data_index_sampled[start], 0),
                            data_index_sampled[end - 1] - data_index_sampled[start],
                            ratio,
                            facecolor=sns.color_palette("tab10")[
                                (color_offset + motif_set_count + aaa) %
                                len(sns.color_palette("tab10"))],
                            alpha=0.7
                        )

                        rx, ry = rect.get_xy()
                        cx = rx + rect.get_width() / 2.0
                        cy = ry + rect.get_height() / 2.0
                        axes[1, 0].annotate(column, (cx, cy),
                                            color='black',
                                            weight='bold',
                                            fontsize=12,
                                            ha='center',
                                            va='center')

                        axes[1, 0].add_patch(rect)

        if ground_truth is not None and len(ground_truth) > 0:
            gt_count = 1
            y_labels.append("Ground Truth")

        if motifsets is not None:
            for i, motif_set in enumerate(motifsets_sampled):
                if motif_set is not None:
                    motif_length_sampled = np.int32(max(2, motif_length // factor))

                    for pos in motif_set:
                        if pos + motif_length_sampled - 1 < dim_raw_sampled.shape[0]:
                            ratio = 0.8
                            rect = Rectangle(
                                (data_index_sampled[pos], -i - gt_count),
                                data_index_sampled[pos + motif_length_sampled - 1] -
                                data_index_sampled[pos],
                                ratio,
                                facecolor=sns.color_palette("tab10")[
                                    (color_offset + i) % len(
                                        sns.color_palette("tab10"))],
                                alpha=0.7
                            )
                            axes[1, 0].add_patch(rect)

                    # label = (("Motif Set " + str(i + 1)))
                    if dist is not None:
                        label = "Motif Set, k=" + str(len(motifsets)) + ", d=" + str(
                            np.round(dist, 2))
                    else:
                        label = "Motif Set, k=" + str(len(motifsets))

                        y_labels.append(label)

    if len(y_labels) > 0:
        axes[1, 0].set_yticks(-np.arange(len(y_labels)) + 0.5)
        axes[1, 0].set_yticklabels(y_labels, fontsize=18)
        axes[1, 0].set_ylim([-abs(len(y_labels)) + 1, 1])
        axes[1, 0].set_xlim(axes[0, 0].get_xlim())
        axes[1, 0].set_xticklabels([])
        axes[1, 0].set_xticks([])

        if motifsets is not None:
            axes[1, 0].set_title("Positions", fontsize=22)

        for i in range(1, axes.shape[-1]):
            axes[1, i].remove()

    if isinstance(data, pd.DataFrame):
        axes[0, 0].set_yticks(tick_offsets)
        axes[0, 0].set_yticklabels(data.index, fontsize=18)
        axes[0, 0].set_xlabel("Time", fontsize=18)

        if motifsets is not None:
            axes[0, 1].set_yticks(tick_offsets)
            axes[0, 1].set_yticklabels(data.index, fontsize=18)
            axes[0, 1].set_xlabel("Length", fontsize=18)

    sns.despine()
    fig.tight_layout()

    if show:
        plt.show()

    return fig, axes


def _plot_elbow_points(
        ds_name, data, motif_length,
        elbow_points,
        motifset_candidates,
        dists):
    """Plots the elbow points found.

    Parameters
    ----------
    ds_name: String
        The name of the time series.
    data: array-like
        The time series data.
    motif_length: int
        The length of the motif.
    elbow_points: array-like
        The elbow points to plot.
    motifset_candidates: 2d array-like
        The motifset candidates. Will only extract those motif sets
        within elbow_points.
    dists: array-like
        The distances (extents) for each motif set
    """

    data_index, data_raw = ml.pd_series_to_numpy(data)

    if data_raw.ndim == 1:
        data_raw = data_raw.reshape((1, -1))

    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.set_title(ds_name + "\nElbow Points")
    ax.plot(range(2, len(np.sqrt(dists))), dists[2:], "b", label="Extent")

    lim1 = plt.ylim()[0]
    lim2 = plt.ylim()[1]
    for elbow in elbow_points:
        ax.vlines(
            elbow, lim1, lim2,
            linestyles="--", label=str(elbow) + "-Motiflet"
        )
    ax.set(xlabel='Size (k)', ylabel='Extent')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    motiflets = motifset_candidates[elbow_points]
    for i, motiflet in enumerate(motiflets):
        if motiflet is not None:
            axins = ax.inset_axes(
                [(elbow_points[i] - 3) / (len(motifset_candidates) - 2), 0.7, 0.3, 0.3])

            df = pd.DataFrame()
            df["time"] = data_index[range(0, motif_length)]
            for aa, pos in enumerate(motiflet):
                # Shows only first dimension
                df[str(aa)] = zscore(data_raw[0, pos:pos + motif_length])

            df_melt = pd.melt(df, id_vars="time")

            _ = sns.lineplot(ax=axins, data=df_melt, x="time", y="value", ci=99,
                             n_boot=10, color=sns.color_palette("tab10")[i % 10])
            axins.set_xlabel("")
            axins.set_ylabel("")
            axins.xaxis.set_major_formatter(plt.NullFormatter())
            axins.yaxis.set_major_formatter(plt.NullFormatter())

    plt.show()


def plot_elbow(
        k_max,
        data,
        ds_name,
        motif_length,
        plot_elbows=False,
        plot_grid=True,
        ground_truth=None,
        method_name=None,
        filter=True,
        n_jobs=4,
        elbow_deviation=1.00,
        slack=0.5,
        distance=znormed_euclidean_distance,
        distance_single=znormed_euclidean_distance_single,
        distance_preprocessing=sliding_mean_std,
        backend="scalable"
):
    """Plots the elbow-plot for k-Motiflets.

    This is the method to find and plot the characteristic k-Motiflets within range
    [2...k_max] for given a `motif_length` using elbow-plots.

    Details are given within the paper Section 5.1 Learning meaningful k.

    Parameters
    ----------
    k_max: int
        use [2...k_max] to compute the elbow plot (user parameter).
    data: array-like
        the TS
    ds_name: String
        the name of the dataset
    motif_length: int
        the length of the motif (user parameter)
    plot_elbows: bool (default=False)
        plots the elbow points into the plot
    ground_truth: pd.Series (default=None)
        Ground-truth information as pd.Series.
    filter: bool (default=True)
        filters overlapping motiflets from the result,
    n_jobs : int (default=4)
        Number of jobs to be used.
    elbow_deviation : float (default=1.00)
        The minimal absolute deviation needed to detect an elbow.
        It measures the absolute change in deviation from k to k+1.
        1.05 corresponds to 5% increase in deviation.
    distance: callable (default=znormed_euclidean_distance)
        The distance function to be computed.
    distance_preprocessing: callable (default=sliding_mean_std)
        The distance preprocessing function to be computed.
    backend : String, default="scalable"
        The backend to use. As of now 'scalable', 'sparse' and 'default' are supported.
        Use 'default' for the original exact implementation with excessive memory,
        Use 'scalable' for a scalable, exact implementation with less memory,
        Use 'sparse' for a scalable, exact implementation with more memory.

    Returns
    -------
    Tuple
        dists:          distances for each k in [2...k_max]
        candidates:     motifset-candidates for each k
        elbow_points:   elbow-points

    """
    # turn into 2d array
    if data.ndim == 1:
        if isinstance(data, pd.Series):
            data = data.to_frame().T
        elif isinstance(data, (np.ndarray, np.generic)):
            data = data.reshape(1, -1)

    _, raw_data = ml.pd_series_to_numpy(data)

    startTime = time.perf_counter()
    dists, candidates, elbow_points, m, memory_usage = ml.search_k_motiflets_elbow(
        k_max,
        raw_data,
        motif_length,
        n_jobs=n_jobs,
        elbow_deviation=elbow_deviation,
        slack=slack,
        distance=distance,
        distance_single=distance_single,
        distance_preprocessing=distance_preprocessing,
        backend=backend)
    endTime = (time.perf_counter() - startTime)

    # print(f"Found motiflets in {np.round(endTime, 1)} s")

    if filter:
        elbow_points = ml.filter_unique(elbow_points, candidates, motif_length)

    # print("\tElbow Points", elbow_points)

    if plot_elbows:
        _plot_elbow_points(
            ds_name, data,
            motif_length, elbow_points,
            candidates, dists)

    if plot_grid:
        if data.shape[0] == 1:
            plot_grid_motiflets(
                ds_name,
                data,
                candidates,
                elbow_points,
                dists,
                motif_length,
                method_name=method_name,
                show_elbows=False,
                font_size=24,
                ground_truth=ground_truth)
        else:
            plot_motifset(
                ds_name,
                data,
                motifsets=candidates[elbow_points],
                motif_length=motif_length,
                ground_truth=ground_truth,
                show=True)

    return dists, candidates, elbow_points, memory_usage


def plot_motif_length_selection(
        k_max,
        data,
        motif_length_range,
        ds_name,
        n_jobs=4,
        elbow_deviation=1.00,
        slack=0.5,
        subsample=2,
        distance=znormed_euclidean_distance,
        distance_single=znormed_euclidean_distance_single,
        distance_preprocessing=sliding_mean_std,
        backend="scalable"
):
    """Computes the AU_EF plot to extract the best motif lengths

    This is the method to find and plot the characteristic motif-lengths, for k in
    [2...k_max], using the area AU-EF plot.

    Details are given within the paper 5.2 Learning Motif Length l.

    Parameters
    ----------
    k_max: int
        use [2...k_max] to compute the elbow plot.
    data: array-like
        the TS
    motif_length_range: array-like
        the interval of lengths
    ds_name: String
        Name of the time series for displaying
    n_jobs : int (default=4)
        Number of jobs to be used.
    elbow_deviation : float, default=1.00
        The minimal absolute deviation needed to detect an elbow.
        It measures the absolute change in deviation from k to k+1.
        1.05 corresponds to 5% increase in deviation.
    slack: float (default=0.5)
        Defines an exclusion zone around each subsequence to avoid trivial matches.
        Defined as percentage of m. E.g. 0.5 is equal to half the window length.
    subsample: int (default=2)
        the subsample factor
    distance: callable (default=znormed_euclidean_distance)
        The distance function to be computed.
    distance_preprocessing: callable (default=sliding_mean_std)
        The distance preprocessing function to be computed.
    backend : String, default="scalable"
        The backend to use. As of now 'scalable', 'sparse' and 'default' are supported.
        Use 'default' for the original exact implementation with excessive memory,
        Use 'scalable' for a scalable, exact implementation with less memory,
        Use 'sparse' for a scalable, exact implementation with more memory.

    Returns
    -------
    best_motif_length: int
        The motif length that maximizes the AU-EF.

    """
    # turn into 2d array
    data = convert_to_2d(data)
    index, data_raw = ml.pd_series_to_numpy(data)

    header = " in " + data.index.name if isinstance(
        data, pd.Series) and data.index.name != None else ""

    # discretizes ranges
    motif_length_range = np.int32(motif_length_range)

    startTime = time.perf_counter()
    best_motif_length, _, au_ef, elbow, top_motiflets, _ = \
        ml.find_au_ef_motif_length(
            data_raw, k_max,
            motif_length_range=motif_length_range,
            n_jobs=n_jobs,
            elbow_deviation=elbow_deviation,
            slack=slack,
            subsample=subsample,
            distance=distance,
            distance_single=distance_single,
            distance_preprocessing=distance_preprocessing,
            backend=backend)
    endTime = (time.perf_counter() - startTime)
    print("\tTime", np.round(endTime, 1), "s")
    indices = ~np.isinf(au_ef)
    fig, ax = plt.subplots(figsize=(5, 2))
    ax = sns.lineplot(
        x=motif_length_range[indices],
        y=au_ef[indices],
        label="AU_EF",
        ci=None, estimator=None)
    sns.despine()
    plt.tight_layout()
    ax.set_title("Best length on " + ds_name, size=20)
    ax.set(xlabel='Motif Length' + header, ylabel='Area under EF\n(lower is better)')

    for item in ([ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(16)

    # plt.legend(loc="best")
    fig.set_figheight(5)
    fig.set_figwidth(5)
    plt.show()

    return best_motif_length


def plot_grid_motiflets(
        ds_name, data, motifsets, elbow_points, dist,
        motif_length, font_size=20,
        ground_truth=None,
        method_name=None,
        method_names=None,
        show_elbows=False,
        color_palette=sns.color_palette("tab10"),
        grid_dim=None,
        plot_index=None):
    """Plots the characteristic motifs for each method along the time series.

    Parameters
    ----------
    ds_name: String
        The name of the time series
    data: array-like
        The time series data
    motifsets: 2d array-like
        The motifset candidates
    elbow_points: array-like
        The elbow points found. Only motif sets from the elbow points will be plotted.
    dist: array-like
        The distances (extents) of the motif set candidates
    motif_length: int
        The motif length found.
    font_size: int
        Font-size to use for plotting.
    ground_truth: pd.Series
        Ground-truth information as pd.Series.
    method_name: String
        Name of a single method
    method_names: array-like
        Name of all methods
    show_elbows: bool
        Show an elbow plot
    color_palette:
        Color-palette to use
    grid_dim: int
        The dimensionality of the grid (number of columns)
    plot_index: int
        Plots only the passed methods in the given order

    """

    sns.set(font_scale=2)
    sns.set_style("white")
    sns.set_context("paper",
                    rc={"font.size": font_size,
                        "axes.titlesize": font_size - 8,
                        "axes.labelsize": font_size - 8,
                        "xtick.labelsize": font_size - 10,
                        "ytick.labelsize": font_size - 10, })

    label_cols = 2

    count_plots = 3 if len(motifsets[elbow_points]) > 6 else 2
    if show_elbows:
        count_plots = count_plots + 1

    if ground_truth is None:
        ground_truth = []

    if grid_dim is None:
        if plot_index is not None:
            ll = len(plot_index)
        else:
            ll = len(elbow_points)
        grid_dim = int(max(2, np.ceil(ll / 2)))

    dims = int(np.ceil(len(elbow_points) / grid_dim)) + count_plots

    fig = plt.figure(constrained_layout=True, figsize=(10, dims * 2))
    gs = fig.add_gridspec(dims, grid_dim, hspace=0.8, wspace=0.4)

    ax_ts = fig.add_subplot(gs[0, :])
    ax_ts.set_title("(a) Dataset: " + ds_name + "")

    # turn into 2d array
    if data.ndim == 1:
        if isinstance(data, pd.Series):
            data = data.to_frame().T
        elif isinstance(data, (np.ndarray, np.generic)):
            data = data.reshape(1, -1)

    data_index, data_raw = ml.pd_series_to_numpy(data)
    data_raw_sampled, data_index_sampled = data_raw, data_index

    factor = 1
    if data_raw.shape[-1] > 2_000:
        data_raw_sampled = np.zeros((data_raw.shape[0], 2_000))
        for i in range(data_raw.shape[0]):
            index = MinMaxLTTBDownsampler().downsample(
                np.ascontiguousarray(data_raw[i]), n_out=2_000)
            data_raw_sampled[i] = data_raw[i, index]

        data_index_sampled = data_index[index]
        factor = max(1, data_raw.shape[-1] / data_raw_sampled.shape[-1])
        if motifsets is not None:
            motifsets_sampled = np.array(
                list(map(lambda x: (x // factor) if x is not None else x, motifsets)),
                dtype=np.object_)
    else:
        motifsets_sampled = motifsets

    _ = sns.lineplot(x=data_index_sampled, y=data_raw_sampled[0], ax=ax_ts, linewidth=1)
    sns.despine()

    for aaa, column in enumerate(ground_truth):
        for offsets in ground_truth[column]:
            for pos, offset in enumerate(offsets):
                start = np.int32(offset[0] // factor)
                end = np.int32(offset[1] // factor)
                if pos == 0:
                    sns.lineplot(x=data_index_sampled[start:end],
                                 y=data_raw_sampled[0, start:end],
                                 label=column,
                                 color=color_palette[aaa + 1],
                                 ci=None, estimator=None
                                 )
                else:
                    sns.lineplot(x=data_index_sampled[start:end],
                                 y=data_raw_sampled[0, start:end],
                                 color=color_palette[aaa + 1],
                                 ci=None, estimator=None
                                 )

    if len(motifsets[elbow_points]) > 6:
        ax_bars = fig.add_subplot(gs[1:3, :], sharex=ax_ts)
        next_id = 3
    else:
        ax_bars = fig.add_subplot(gs[1, :], sharex=ax_ts)
        next_id = 2

    ax_bars.set_title("(b) Position of Top Motif Sets")

    if show_elbows:
        ax_elbow = fig.add_subplot(gs[next_id, :])
        ax_elbow.set_title("(c) Significant Elbow Points on " + ds_name)
        ax_elbow.plot(range(len(np.sqrt(dist))), dist, "b", label="Extent")
        lim1 = plt.ylim()[0]
        lim2 = plt.ylim()[1]
        for elbow in elbow_points:
            ax_elbow.vlines(
                elbow, lim1, lim2,
                label=str(elbow) + "-Motiflet"
            )
        ax_elbow.set(xlabel='Size (k)', ylabel='Extent')
        ax_elbow.xaxis.set_major_locator(MaxNLocator(integer=True))

    gs = fig.add_gridspec(dims, grid_dim)

    #### Hack to add a subplot title
    ax_title = fig.add_subplot(gs[count_plots, :])

    if (show_elbows):
        ax_title.set_title('(d) Shape of Top Motif Sets', pad=30)
    else:
        ax_title.set_title('(c) Shape of Top Motif Sets', pad=30)

    # Turn off axis lines and ticks of the big subplot 
    ax_title.tick_params(labelcolor=(1., 1., 1., 0.0),
                         top='off', bottom='off', left='off', right='off')
    ax_title.axis('off')
    ax_title._frameon = False
    sns.despine()
    ######

    # hatches = ['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*']

    y_labels = []
    ii = -1
    print("Debug", elbow_points)
    motiflets_sampled = motifsets_sampled[elbow_points]
    motiflets = motifsets[elbow_points]
    for i, motiflet in enumerate(motiflets_sampled):
        if motiflet is not None:
            motif_length_sampled = np.int32(max(2, motif_length // factor))

            plot_minature = (plot_index == None) or (i in plot_index)
            if plot_minature:
                ii = ii + 1
                off = int(ii / grid_dim)
                ax_motiflet = fig.add_subplot(gs[count_plots + off, ii % grid_dim])

            df = pd.DataFrame()
            df["time"] = data_index[range(0, motif_length)]

            for aa, pos in enumerate(motiflet):
                pos = np.int32(pos)
                ratio = 0.8
                rect = Rectangle(
                    (data_index_sampled[pos], -i),
                    data_index_sampled[pos + motif_length_sampled - 1] -
                    data_index_sampled[pos],
                    ratio,
                    facecolor=color_palette[
                        (len(ground_truth) + ii % grid_dim) % len(color_palette)],
                    # hatch=hatches[i],
                    alpha=0.7
                )
                ax_bars.add_patch(rect)

            for aa, pos in enumerate(motiflets[i]):
                df[str(aa)] = zscore(data_raw[0, pos:pos + motif_length])

            if method_name is not None:
                y_labels.append(method_name + "\nTop-" + str(i + 1))

            elif method_names is not None:
                y_labels.append(method_names[i])

            dists = ""
            if (dist is not None):
                dist = np.array(dist)
                dist[dist == float("inf")] = 0
                dists = str(dist[elbow_points[i]].astype(int))
                # dists = str(int(dist[elbow_points[i]]))

            label = ""
            # if method_names is not None:
            #    label =  method_names[elbow_points[i]]

            if plot_minature:
                df_melt = pd.melt(df, id_vars="time")
                _ = sns.lineplot(ax=ax_motiflet,
                                 data=df_melt,
                                 x="time", y="value",
                                 ci=99, n_boot=10,
                                 color=color_palette[
                                     (len(ground_truth) + ii % grid_dim) % len(
                                         color_palette)],
                                 label=label + "k=" + str(len(motiflet)) + ",d=" + dists
                                 )
                ax_motiflet.set_ylabel("")

                if isinstance(data, pd.Series):
                    ax_motiflet.set_xlabel(data.index.name)

                sns.despine()
                ax_motiflet.legend(loc="upper right")

            if method_names is not None:
                ax_bars.plot([], [], label=method_names[elbow_points[i]].split()[0],
                             linewidth=10,
                             color=color_palette[
                                 (len(ground_truth) + ii % grid_dim) % len(
                                     color_palette)])
                if plot_minature:
                    ax_motiflet.set_title(method_names[elbow_points[i]])

            elif method_name is not None:
                ax_bars.plot([], [], label=method_name, linewidth=10,
                             color=color_palette[
                                 (len(ground_truth) + ii % grid_dim) % len(
                                     color_palette)])
                if plot_minature:
                    ax_motiflet.set_title(method_name + " Top-" + str(i + 1))

            if show_elbows:
                axins = ax_elbow.inset_axes(
                    [elbow_points[i] / len(motifsets), 0.7, 0.1, 0.2])

                _ = sns.lineplot(ax=axins, data=df_melt, x="time", y="value",
                                 ci=0, n_boot=10,
                                 color=color_palette[
                                     (len(ground_truth) + ii % grid_dim) % len(
                                         color_palette)])
                axins.set_xlabel("")
                axins.set_ylabel("")
                axins.xaxis.set_major_formatter(plt.NullFormatter())
                axins.yaxis.set_major_formatter(plt.NullFormatter())

            if plot_minature:
                ax_motiflet.set_yticks([])

    ax_bars.set_yticks(-np.arange(len(y_labels)) + 0.5, )
    ax_bars.set_yticklabels(y_labels, fontsize=12)
    ax_bars.set_ylim([-len(motiflets) + 1, 1])
    # ax_bars.legend(loc="best")

    if (ground_truth is not None and len(ground_truth) > 0):
        ax_ts.legend(loc="upper left", ncol=label_cols)

    plt.tight_layout()
    gs.tight_layout(fig)
    plt.show()


def plot_all_competitors(
        data,
        ds_name,
        motifsets,
        motif_length,
        method_names=None,
        ground_truth=None,
        plot_index=None,
        color_palette=sns.color_palette("tab10"),
        slack=0.5):
    """Plots the found motif sets of multiple competitor methods

    Parameters
    ----------
    ds_name: String
        The name of the time series
    data: array-like
        The time series data
    motifsets: 2d array-like
        The found motif sets for plotting
    motif_length: int
        The motif length found.
    method_names: array-like
        Names of the method to plot
    ground_truth: pd.Series
        Ground-truth information as pd.Series.
    grid_dim: int
        The dimensionality of the grid (number of columns)
    plot_index: int
        Plots only the passed methods in the given order
    """

    # convert to numpy array
    _, data_raw = ml.pd_series_to_numpy(data)
    D_full = ml.compute_distances_full(data_raw, motif_length, slack=slack)
    indices = np.arange(len(motifsets))

    dists = [ml.get_pairwise_extent(D_full, motiflet_pos, upperbound=np.inf)
             for motiflet_pos in motifsets]

    plot_grid_motiflets(
        ds_name, data, motifsets, indices,
        dists, motif_length,
        font_size=26,
        method_names=method_names,
        ground_truth=ground_truth,
        color_palette=color_palette,
        plot_index=plot_index)


def plot_competitors(
        data,
        ds_name,
        motifsets,
        motif_length,
        prefix="",
        filter=True,
        ground_truth=None,
        slack=0.5):
    """Plots motif sets of a single competitor method.

    Parameters
    ----------
    data: array-like
        The time series data
    ds_name: String
        The name of the time series
    motifsets: array-like
        The motifset for plotting
    motif_length: int
        The motif length found.
    prefix: String
        The method name
    filter: bool, default=True
        filter overlapping motifs
    ground_truth: pd.Series
        Ground-truth information as pd.Series.

    """

    # convert to numpy array
    _, data_raw = ml.pd_series_to_numpy(data)

    D_full = ml.compute_distances_full(data_raw, motif_length, slack=slack)

    last = -1
    motifsets_filtered = []
    for motifset in motifsets:
        if ((len(motifset) > last) or (not filter)):
            motifsets_filtered.append(motifset)
            last = len(motifset)
    motifsets_filtered = np.array(motifsets_filtered)

    elbow_points = np.arange(len(motifsets_filtered))

    if filter:
        elbow_points = ml.filter_unique(elbow_points, motifsets_filtered, motif_length)

    dists = [ml.get_pairwise_extent(D_full, motiflet_pos, upperbound=np.inf)
             for motiflet_pos in motifsets_filtered]

    plot_grid_motiflets(
        ds_name, data, motifsets_filtered, elbow_points,
        dists, motif_length, method_name=prefix,
        ground_truth=ground_truth)

    return motifsets_filtered[elbow_points]


def format_key(e):
    key = ""
    if e > 0:
        key = "+" + str(e * 100) + "%"
    elif e < 0:
        key = str(e * 100) + "%"
    return key


def to_df(motif_sets, method_name, df, df2=None):
    df_all_1 = pd.DataFrame()
    df_all_2 = pd.DataFrame()
    for key in motif_sets:
        ms_set_finder = motif_sets[key]
        df_all_1[method_name + " Top-1 " + key] = [ms_set_finder[-1]]
        df[method_name + " Top-1 " + key] = [ms_set_finder[-1]]

        if df2 is not None:
            df_all_2[method_name + " Top-2 " + key] = [ms_set_finder[-2]]
            df2[method_name + " Top-2 " + key] = [ms_set_finder[-2]]

    if df2 is not None:
        df_all = (pd.concat([df_all_1, df_all_2], axis=1)).T
    else:
        df_all = df_all_1.T

    df_all.rename(columns={0: "offsets"}, inplace=True)
    return df_all
