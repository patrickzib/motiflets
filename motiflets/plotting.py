# -*- coding: utf-8 -*-
"""Plotting utilities.
"""

__author__ = ["patrickzib"]

import time

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
from scipy.stats import zscore

import motiflets.motiflets as ml

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


def plot_dataset(
        ds_name,
        data,
        ground_truth=None
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
    plain=False

    """
    return plot_motifset(ds_name, data, ground_truth=ground_truth)


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
        motifset=None,
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
    motifset: array like
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

    if motifset is not None:
        fig, axes = plt.subplots(1, 2, sharey=False,
                                 sharex=False, figsize=(20, 3),
                                 gridspec_kw={'width_ratios': [4, 1]})
    else:
        fig, axes = plt.subplots(1, 1, figsize=(20, 3))
        axes = [axes]

    if ground_truth is None:
        ground_truth = []

    data_index, data_raw = ml.pd_series_to_numpy(data)

    axes[0].set_title(ds_name, fontsize=20)
    _ = sns.lineplot(x=data_index, y=data_raw, ax=axes[0], linewidth=1,
                     ci=None, estimator=None)
    sns.despine()

    if motifset is not None:
        for pos in motifset:
            _ = sns.lineplot(ax=axes[0],
                             x=data_index[np.arange(pos, pos + motif_length)],
                             y=data_raw[pos:pos + motif_length], linewidth=5,
                             color=sns.color_palette("tab10")[
                                 (len(ground_truth) + 2) % 10],
                             # alpha=0.5,
                             ci=None, estimator=None)

    for aaa, column in enumerate(ground_truth):
        for offsets in ground_truth[column]:
            for pos, offset in enumerate(offsets):
                if pos == 0:
                    sns.lineplot(x=data_index[offset[0]: offset[1]],
                                 y=data_raw[offset[0]:offset[1]],
                                 label=column,
                                 color=sns.color_palette("tab10")[(aaa + 1) % 10],
                                 ax=axes[0],
                                 ci=None, estimator=None
                                 )
                else:
                    sns.lineplot(x=data_index[offset[0]: offset[1]],
                                 y=data_raw[offset[0]:offset[1]],
                                 color=sns.color_palette("tab10")[(aaa + 1) % 10],
                                 ax=axes[0],
                                 ci=None, estimator=None
                                 )

    if motifset is not None:
        axes[1].set_title(
            "Motif Set, k=" + str(len(motifset)) + ", d=" + str(np.round(dist, 2)),
            fontsize=20)

        df = pd.DataFrame()
        df["time"] = np.arange(0, motif_length)

        for aa, pos in enumerate(motifset):
            df[str(aa)] = zscore(data_raw[pos:pos + motif_length])

        df_melt = pd.melt(df, id_vars="time")
        _ = sns.lineplot(ax=axes[1], data=df_melt, ci=99, n_boot=10,
                         x="time", y="value")

    sns.despine()

    if motifset is None:
        motifset = []

    fig.tight_layout()
    if show:
        plt.show()

    return fig, axes


def _plot_elbow_points(
        ds_name, data, motif_length, elbow_points,
        motifset_candidates, dists):
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
                df[str(aa)] = zscore(data_raw[pos:pos + motif_length])

            df_melt = pd.melt(df, id_vars="time")

            _ = sns.lineplot(ax=axins, data=df_melt, x="time", y="value", ci=99,
                             n_boot=10, color=sns.color_palette("tab10")[i % 10])
            axins.set_xlabel("")
            axins.set_ylabel("")
            axins.xaxis.set_major_formatter(plt.NullFormatter())
            axins.yaxis.set_major_formatter(plt.NullFormatter())

    plt.show()


def plot_elbow(k_max,
               data,
               ds_name,
               motif_length,
               exclusion=None,
               plot_elbows=False,
               ground_truth=None,
               filter=True,
               method_name=None,
               elbow_deviation=1.00,
               slack=0.5):
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
    exclusion: 2d-array
        exclusion zone - use when searching for the TOP-2 motiflets
    plot_elbows: bool, default=False
        plots the elbow ploints into the plot
    ground_truth: pd.Series
        Ground-truth information as pd.Series.
    filter: bool, default=True
        filters overlapping motiflets from the result,
    method_name:  String
        used for display only.
    elbow_deviation : float, default=1.00
        The minimal absolute deviation needed to detect an elbow.
        It measures the absolute change in deviation from k to k+1.
        1.05 corresponds to 5% increase in deviation.

    Returns
    -------
    Tuple
        dists:          distances for each k in [2...k_max]
        candidates:     motifset-candidates for each k
        elbow_points:   elbow-points

    """
    _, raw_data = ml.pd_series_to_numpy(data)
    print("Data", len(raw_data))

    startTime = time.perf_counter()
    dists, candidates, elbow_points, m = ml.search_k_motiflets_elbow(
        k_max,
        raw_data,
        motif_length,
        exclusion=exclusion,
        elbow_deviation=elbow_deviation,
        slack=slack)
    endTime = (time.perf_counter() - startTime)

    print("Chosen window-size:", m, "in", np.round(endTime, 1), "s")

    if filter:
        elbow_points = ml._filter_unique(elbow_points, candidates, motif_length)

    print("Elbow Points", elbow_points)

    if plot_elbows:
        _plot_elbow_points(ds_name, data, motif_length, elbow_points, candidates, dists)

    print("Data", len(data))

    plot_grid_motiflets(
        ds_name, data, candidates, elbow_points,
        dists, motif_length, show_elbows=plot_elbows,
        font_size=24,
        ground_truth=ground_truth, method_name=method_name)

    return dists, candidates, elbow_points


def plot_motif_length_selection(
        k_max, data, motif_length_range, ds_name,
        elbow_deviation=1.00, slack=0.5):
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
    elbow_deviation : float, default=1.00
        The minimal absolute deviation needed to detect an elbow.
        It measures the absolute change in deviation from k to k+1.
        1.05 corresponds to 5% increase in deviation.

    Returns
    -------
    best_motif_length: int
        The motif length that maximizes the AU-PDF.

    """
    index, _ = ml.pd_series_to_numpy(data)
    header = " in " + data.index.name if isinstance(data,
                                                    pd.Series) and data.index.name != None else ""

    # discretizes ranges
    motif_length_range = np.int32(motif_length_range)

    startTime = time.perf_counter()
    best_motif_length, au_ef, elbow, top_motiflets = \
        ml.find_au_ef_motif_length(
            data, k_max,
            motif_length_range=motif_length_range,
            elbow_deviation=elbow_deviation,
            slack=slack)
    endTime = (time.perf_counter() - startTime)
    print("\tTime", np.round(endTime, 1), "s")

    indices = ~np.isinf(au_ef)
    fig, ax = plt.subplots(figsize=(5, 2))
    ax = sns.lineplot(
        x=index[motif_length_range[indices]],
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
        ds_name, data, candidates, elbow_points, dist,
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
    candidates: 2d array-like
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

    count_plots = 3 if len(candidates[elbow_points]) > 6 else 2
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

    data_index, data_raw = ml.pd_series_to_numpy(data)

    _ = sns.lineplot(x=data_index, y=data_raw, ax=ax_ts, linewidth=1)
    sns.despine()

    for aaa, column in enumerate(ground_truth):
        for offsets in ground_truth[column]:
            for pos, offset in enumerate(offsets):
                if pos == 0:
                    sns.lineplot(x=data_index[offset[0]: offset[1]],
                                 y=data_raw[offset[0]:offset[1]],
                                 label=column,
                                 color=color_palette[aaa + 1],
                                 ci=None, estimator=None
                                 )
                else:
                    sns.lineplot(x=data_index[offset[0]: offset[1]],
                                 y=data_raw[offset[0]:offset[1]],
                                 color=color_palette[aaa + 1],
                                 ci=None, estimator=None
                                 )

    if len(candidates[elbow_points]) > 6:
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
        ax_title.set_title('(d) Shape of Top Motif Sets by Method', pad=30)
    else:
        ax_title.set_title('(c) Shape of Optimal Motif Sets by Method', pad=30)

    # Turn off axis lines and ticks of the big subplot 
    ax_title.tick_params(labelcolor=(1., 1., 1., 0.0),
                         top='off', bottom='off', left='off', right='off')
    ax_title.axis('off')
    ax_title._frameon = False
    sns.despine()
    ######

    hatches = ['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*']

    y_labels = []
    ii = -1
    motiflets = candidates[elbow_points]
    for i, motiflet in enumerate(motiflets):
        if motiflet is not None:

            plot_minature = (plot_index == None) or (i in plot_index)
            if plot_minature:
                ii = ii + 1
                off = int(ii / grid_dim)
                ax_motiflet = fig.add_subplot(gs[count_plots + off, ii % grid_dim])

            df = pd.DataFrame()
            df["time"] = data_index[range(0, motif_length)]

            for aa, pos in enumerate(motiflet):
                df[str(aa)] = zscore(data_raw[pos:pos + motif_length])
                ratio = 0.8
                rect = Rectangle(
                    (data_index[pos], -i),
                    data_index[pos + motif_length - 1] - data_index[pos],
                    ratio,
                    facecolor=color_palette[
                        (len(ground_truth) + ii % grid_dim) % len(color_palette)],
                    # hatch=hatches[i],
                    alpha=0.7
                )
                ax_bars.add_patch(rect)

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
                _ = sns.lineplot(ax=ax_motiflet, data=df_melt,
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
                    [elbow_points[i] / len(candidates), 0.7, 0.1, 0.2])

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
        elbow_points = ml._filter_unique(elbow_points, motifsets_filtered, motif_length)

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
