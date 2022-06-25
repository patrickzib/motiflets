import os

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
from numba import njit
from scipy.stats import zscore

import motiflets.motiflets as ml

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# Store figures??
save_fig = True

def plot_dataset(name, data, ds_name, ground_truth=None):
    plot_motifset(name, data, ds_name=ds_name, ground_truth=ground_truth)


def append_all_motif_sets(df, motif_sets, method_name, D_full):
    filtered_motif_sets = [m for m in motif_sets if m is not None]
    extent = [ml.get_pairwise_extent(D_full, motiflet) for motiflet in
              filtered_motif_sets]
    count = [len(motiflet) for motiflet in filtered_motif_sets]

    for m, e, c in zip(filtered_motif_sets, extent, count):
        entry = {"Method": method_name, "Motif": m, "Extent": e, "k": c}
        df = df.append(entry, ignore_index=True)
    return df


def plot_motifset(
        name, data,
        motifset=None, dist=None,
        motif_length=None,
        idx=None, ds_name=None,
        prefix="",
        ground_truth=None):
    if motifset is not None:
        fig, axes = plt.subplots(1, 2, sharey=False,
                                 sharex=False, figsize=(20, 3),
                                 gridspec_kw={'width_ratios': [4, 1]})
    else:
        fig, axes = plt.subplots(1, 1, figsize=(20, 3))
        axes = [axes]

    if ground_truth is None:
        ground_truth = []

    if isinstance(data, pd.Series):
        data_raw = data.values
        data_index = data.index
    else:
        data_raw = data
        data_index = np.arange(len(data))

    axes[0].set_title(ds_name if ds_name is not None else name, fontsize=20)
    _ = sns.lineplot(x=data_index, y=data_raw, ax=axes[0], linewidth=1)
    sns.despine()

    if motifset is not None:
        for pos in motifset:
            _ = sns.lineplot(ax=axes[0],
                             x=data_index[np.arange(pos, pos + motif_length)],
                             y=data_raw[pos:pos + motif_length], linewidth=5,
                             color=sns.color_palette()[len(ground_truth) + 2])

    if isinstance(data, pd.Series):
        data_raw = data.to_numpy()
    else:
        data_raw = data

    for aaa, column in enumerate(ground_truth):
        for offsets in ground_truth[column]:
            for pos, offset in enumerate(offsets):
                if pos == 0:
                    sns.lineplot(x=data_index[offset[0]: offset[1]],
                                 y=data_raw[offset[0]:offset[1]],
                                 label=column,
                                 color=sns.color_palette()[aaa + 1],
                                 ax=axes[0]
                                 )
                else:
                    sns.lineplot(x=data_index[offset[0]: offset[1]],
                                 y=data_raw[offset[0]:offset[1]],
                                 color=sns.color_palette()[aaa + 1],
                                 ax=axes[0]
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
        _ = sns.lineplot(ax=axes[1], data=df_melt, ci=99, x="time", y="value")

    sns.despine()

    if motifset is None:
        motifset = []

    fig.tight_layout()

    if save_fig:
        dir_name = "../images/"
        if idx is None:
            plt.savefig(
                dir_name + "/" + name + "_" + str(prefix) +
                "_" + str(len(motifset)) + ".pdf", bbox_inches='tight')
        else:
            plt.savefig(
                dir_name + "/" + name + "_" + str(prefix) +
                "_" + str(len(motifset)) + "_" + str(idx) + ".pdf", bbox_inches='tight')

    plt.show()


def plot_elbow_points(
        name, data, motif_length, elbow_points, candidates, dists):
    
    if isinstance(data, pd.Series):
        data_raw = data.values
        data_index = data.index
    else:
        data_raw = data
        data_index = np.arange(len(data))

    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.set_title(name + "\nElbow Points")
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

    motiflets = candidates[elbow_points]
    for i, motiflet in enumerate(motiflets):
        if motiflet is not None:
            axins = ax.inset_axes(
                 [(elbow_points[i]-3) / (len(candidates)-2), 0.7, 0.3, 0.3])

            df = pd.DataFrame()
            df["time"] = data_index[range(0, motif_length)]
            for aa, pos in enumerate(motiflet):
                df[str(aa)] = zscore(data_raw[pos:pos + motif_length])

            df_melt = pd.melt(df, id_vars="time")

            _ = sns.lineplot(ax=axins, data=df_melt, x="time", y="value", ci=99,
                             color=sns.color_palette()[i])
            axins.set_xlabel("")
            axins.set_ylabel("")
            axins.xaxis.set_major_formatter(plt.NullFormatter())
            axins.yaxis.set_major_formatter(plt.NullFormatter())

    if save_fig:
        plt.savefig("../images/" + name.replace(" ", "-") + "_elbow.pdf", bbox_inches='tight')
    plt.show()


def plot_elbow(ks,
               data,
               dataset,
               motif_length,
               ds_name=None,
               exclusion=None,
               idx=None,
               plot_elbows=False,
               ground_truth=None,
               filter=True,
               method_name=None,
               data_array=None):
    raw_data = data.to_numpy() if isinstance(data, pd.Series) else data
    dists, candidates, elbow_points, m = ml.search_k_motiflets_elbow(
        ks,
        raw_data,
        dataset,
        motif_length,
        exclusion=exclusion)

    print("Chosen window-size:", m, "Elbow Points", elbow_points)

    if exclusion is not None and idx is None:
        idx = "top-2"

    if filter:
        elbow_points = filter_unqiue(elbow_points, candidates, motif_length)

    if plot_elbows:
        plot_elbow_points(ds_name, data, motif_length, elbow_points, candidates, dists)

    plot_grid_motiflets(
        dataset, data, candidates, elbow_points,
        dists, motif_length, idx=idx, ds_name=ds_name, show_elbows=False,
        ground_truth=ground_truth, method_name=method_name, data_array=data_array)

    return dists, candidates, elbow_points


def filter_unqiue(elbow_points, candidates, motif_length):
    filtered_ebp = []
    for i in range(len(elbow_points)):
        unique = True
        for j in range(i + 1, len(elbow_points)):
            unique = check_unique(
                candidates[elbow_points[i]], candidates[elbow_points[j]], motif_length)
            if not unique:
                break
        if unique:
            filtered_ebp.append(elbow_points[i], )
    print("Filtered Elbow Points", filtered_ebp)
    return np.array(filtered_ebp)


@njit
def check_unique(elbow_points_1, elbow_points_2, motif_length):
    uniques = True
    count = 0
    for a in elbow_points_1:  # smaller motiflet
        for b in elbow_points_2:  # larger motiflet
            if (abs(a - b) < (motif_length / 8)):
                count = count + 1
                break

        if count >= len(elbow_points_1) / 2:
            return False
    return True


def plot_motif_length_selection(ks, data, dataset, motif_length_range, ds_name):
    # raw_data = data.values if isinstance(data, pd.Series) else data
    index = data.index if isinstance(data, pd.Series) else np.arange(len(data))
    header = " in " + data.index.name if isinstance(data, pd.Series) and data.index.name != None else ""

    best_motif_length, au_pdfs, elbow, top_motiflets = \
        ml.find_au_pef_motif_length(
            data, dataset, ks, motif_length_range=motif_length_range)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax = sns.lineplot(
        x=index[motif_length_range],
        y=au_pdfs,
        label="AU_EF")
    sns.despine()
    # plt.tight_layout()
    ax.set_title("Best length on " + ds_name, size=16)
    ax.set(xlabel='Motif Length' + header, ylabel='Area under EF\n(lower is better)')

    # plt.legend(loc="best")      
    fig.set_figheight(5)
    fig.set_figwidth(5)
    if save_fig:
        plt.savefig(
            "../images/" + ds_name.replace(" ", "-") + "_motif_length_selection.pdf",
            bbox_inches='tight')
    plt.show()

    return best_motif_length


def plot_grid_motiflets(
        name, data, candidates, elbow_points, dist,
        motif_length, font_size=20,
        idx=None, ds_name=None,
        ground_truth=None,
        method_name=None,
        method_names=None,
        show_elbows=False,
        color_palette=sns.color_palette(),
        grid_dim=None,
        plot_index=None,
        data_array=None):
    sns.set_context("paper",
                    rc={"font.size": font_size, "axes.titlesize": font_size - 8,
                        "axes.labelsize": font_size - 8})

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
    gs = fig.add_gridspec(dims, grid_dim, hspace=0.5, wspace=0.4)

    ax_ts = fig.add_subplot(gs[0, :])
    ax_ts.set_title("(a) Dataset: " + (ds_name if ds_name is not None else name) + "")

    if isinstance(data, pd.Series):
        data_raw = data.values
        data_index = data.index
    else:
        data_raw = data
        data_index = np.arange(len(data))

    if data_array is None:
        _ = sns.lineplot(x=data_index, y=data_raw, ax=ax_ts, linewidth=1)

    if data_array is not None:
        last_offset = 0
        for d in data_array:
            d_data = d.values
            d_index = np.arange(last_offset, last_offset+len(d_data))
            _ = sns.lineplot(x=d_index, y=d_data, ax=ax_ts, label=d.name, linewidth=1)
            last_offset = last_offset + len(d_data)

        ax_ts.legend()

    sns.despine()


    for aaa, column in enumerate(ground_truth):
        for offsets in ground_truth[column]:
            for pos, offset in enumerate(offsets):
                if pos == 0:
                    sns.lineplot(x=data_index[offset[0]: offset[1]],
                                 y=data_raw[offset[0]:offset[1]],
                                 label=column,
                                 color=sns.color_palette()[aaa + 1],
                                 )
                else:
                    sns.lineplot(x=data_index[offset[0]: offset[1]],
                                 y=data_raw[offset[0]:offset[1]],
                                 color=sns.color_palette()[aaa + 1],
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
        ax_elbow.set_title("(c) Significant Elbow Points on " + (
            ds_name if ds_name is not None else name))
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
                ii = ii+1
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
                dists = str(int(dist[elbow_points[i]]))

            label = ""
            # if method_names is not None:
            #    label =  method_names[elbow_points[i]]

            if plot_minature:
                df_melt = pd.melt(df, id_vars="time")
                _ = sns.lineplot(ax=ax_motiflet, data=df_melt, x="time", y="value", ci=99,
                                 color=color_palette[
                                     (len(ground_truth) + ii % grid_dim) % len(color_palette)],
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

                _ = sns.lineplot(ax=axins, data=df_melt, x="time", y="value", ci=99,
                                 color=color_palette[
                                     (len(ground_truth) + ii % grid_dim) % len(
                                         color_palette)])
                axins.set_xlabel("")
                axins.set_ylabel("")
                axins.xaxis.set_major_formatter(plt.NullFormatter())
                axins.yaxis.set_major_formatter(plt.NullFormatter())

            if plot_minature:
                ax_motiflet.set_yticks([])

    ax_bars.set_yticks(-np.arange(len(y_labels)) + 0.5)
    ax_bars.set_yticklabels(y_labels)
    ax_bars.set_ylim([-len(motiflets)+1, 1])
    # ax_bars.legend(loc="best")

    if (ground_truth is not None and len(ground_truth) > 0):
        ax_ts.legend(loc="upper left", ncol=label_cols)

    plt.tight_layout()
    gs.tight_layout(fig)

    if method_names is not None and save_fig:
        if idx is None:
            plt.savefig("../images/" + name.replace(" ", "-") + "_grid.pdf",
                        bbox_inches='tight')
        else:
            plt.savefig(
                "../images/" + name.replace(" ", "-") + "_grid_" + str(idx) + ".pdf",
                bbox_inches='tight')

    if save_fig:
        plt.savefig("../images/" + name.replace(" ", "-") + "_elbows.pdf",
                    bbox_inches='tight')

    plt.show()


def plot_all_competitors(
        data,
        name,
        motifsets,
        motif_length,
        method_names=None,
        target_cardinalities=None,
        ground_truth=None,
        plot_index=None,
        color_palette=sns.color_palette()):
    # convert to numpy array
    data_raw = data
    if isinstance(data, pd.Series):
        data_raw = data.to_numpy()

    D_full = ml.compute_distances_full(data_raw, motif_length)
    indices = np.arange(len(motifsets))

    dists = [ml.get_pairwise_extent(D_full, motiflet_pos, upperbound=np.inf)
             for motiflet_pos in motifsets]

    plot_grid_motiflets(
        name, data, motifsets, indices,
        dists, motif_length, ds_name=name,
        # method_name=prefix,
        method_names=method_names,
        ground_truth=ground_truth,
        color_palette=color_palette,
        plot_index=plot_index)


def plot_competitors(
        data,
        name,
        motifsets,
        motif_length,
        prefix="",
        filter=True,
        target_cardinalities=None,
        ground_truth=None):
    # convert to numpy array
    data_raw = data
    if isinstance(data, pd.Series):
        data_raw = data.to_numpy()

    D_full = ml.compute_distances_full(data_raw, motif_length)

    # max radius
    """
    for elem in motifsets:
        if len(elem) > 1:
            print("r:", ml.get_radius(D_full, elem),
                  "d:", ml.get_pairwise_extent(D_full, elem, upperbound=np.inf))
    """

    last = -1
    motifsets_filtered = []
    for motifset in motifsets:
        if ((len(motifset) > last) or (not filter)):
            motifsets_filtered.append(motifset)
            last = len(motifset)
    motifsets_filtered = np.array(motifsets_filtered)

    elbow_points = np.arange(len(motifsets_filtered))

    if filter:
        elbow_points = filter_unqiue(elbow_points, motifsets_filtered, motif_length)

    dists = [ml.get_pairwise_extent(D_full, motiflet_pos, upperbound=np.inf)
             for motiflet_pos in motifsets_filtered]

    plot_grid_motiflets(
        name, data, motifsets_filtered, elbow_points,
        dists, motif_length, ds_name=name, method_name=prefix,
        ground_truth=ground_truth)

    return motifsets_filtered[elbow_points]

def format_key(e):
    key = ""
    if e > 0:
        key = "+"+str(e*100) + "%"
    elif e < 0:
        key = str(e*100) + "%"
    return key