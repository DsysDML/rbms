from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def plot_scatter_labels(ax, data_proj, gen_data_proj, proj1, proj2, labels):
    """Args:
    ax
    data_proj
    gen_data_proj
    proj1
    proj2
    labels
    """
    ax.scatter(
        data_proj[:, proj1],
        data_proj[:, proj2],
        color="black",
        s=50,
        label=labels[0],
        zorder=0,
        alpha=0.3,
    )
    ax.scatter(
        gen_data_proj[:, proj1],
        gen_data_proj[:, proj2],
        color="red",
        label=labels[1],
        s=20,
        zorder=2,
        edgecolor="black",
        marker="o",
        alpha=1,
        linewidth=0.4,
    )


def plot_hist(ax, data_proj, gen_data_proj, color, proj, labels, orientation="vertical"):
    """Args:
    ax
    data_proj
    gen_data_proj
    color
    proj
    labels
    orientation: (Default value = "vertical")
    """
    ax.hist(
        data_proj[:, proj],
        bins=40,
        color="black",
        histtype="step",
        label=labels[0],
        zorder=0,
        density=True,
        orientation=orientation,
        lw=1,
    )
    ax.hist(
        gen_data_proj[:, proj],
        bins=40,
        color=color,
        histtype="step",
        label=labels[1],
        zorder=1,
        density=True,
        orientation=orientation,
        lw=1.5,
    )
    ax.axis("off")


def plot_PCA(data1, data2, labels, dir1=0, dir2=1):
    """Args:
    data1
    data2
    labels
    dir1: (Default value = 0)
    dir2: (Default value = 1)
    """
    fig = plt.figure(dpi=100, figsize=(5, 5))
    gs = GridSpec(4, 4)

    ax_scatter = fig.add_subplot(gs[1:4, 0:3])
    ax_hist_x = fig.add_subplot(gs[0, 0:3])
    ax_hist_y = fig.add_subplot(gs[1:4, 3])

    plot_scatter_labels(ax_scatter, data1, data2, dir1, dir2, labels=labels)
    plot_hist(ax_hist_x, data1, data2, "red", dir1, labels=labels)
    plot_hist(
        ax_hist_y, data1, data2, "red", dir2, orientation="horizontal", labels=labels
    )

    ax_hist_x.legend(fontsize=12, bbox_to_anchor=(1, 1))
    h, legend = ax_scatter.get_legend_handles_labels()
    ax_scatter.set_xlabel(f"PC{dir1}")
    ax_scatter.set_ylabel(f"PC{dir2}")


def plot_image(
    sample, shape=(28, 28), grid_size=(10, 10), show_grid=False, randomize=True
):
    """Args:
    sample
    shape: (Default value = (28)
    28)
    grid_size: (Default value = (10)
    10)
    show_grid: (Default value = False)
    randomize: (Default value = True)
    """
    num_samples = grid_size[0] * grid_size[1]
    if randomize:
        id_sample = np.random.randint(0, sample.shape[0], num_samples)
    else:
        id_sample = np.arange(num_samples)

    # Create a display array with the appropriate size
    display = np.zeros((shape[0] * grid_size[0], shape[1] * grid_size[1]))

    for i, id_s in enumerate(id_sample):
        # Calculate the row and column for the grid
        idx = i // grid_size[1]  # Row index
        idy = i % grid_size[1]  # Column index

        # Ensure the sample can be reshaped to the specified shape
        display[
            (idx * shape[0]) : ((idx + 1) * shape[0]),
            (idy * shape[1]) : ((idy + 1) * shape[1]),
        ] = sample[id_s].reshape(shape)  # Directly reshape to `shape`

    # Plot the display image
    fig, ax = plt.subplots(1, 1)
    ax.imshow(display, cmap="gray")
    ax.axis("off")  # Hide axes

    if show_grid:
        # Minor ticks for the grid
        ax.set_xticks(np.arange(-0.5, grid_size[1] * shape[1], shape[1]), minor=True)
        ax.set_yticks(np.arange(-0.5, grid_size[0] * shape[0], shape[0]), minor=True)

        # Gridlines based on minor ticks
        ax.grid(which="minor", color="gray", linestyle="-", linewidth=2)


def plot_one_PCA(
    ax: plt.Subplot,
    data1: np.ndarray,
    data2: Optional[np.ndarray] = None,
    labels: Optional[List[str]] = None,
    dir1: int = 0,
    dir2: int = 1,
):
    label_1 = None
    label_2 = None
    if labels is not None:
        label_1 = labels[0]
        if data2 is not None:
            label_2 = labels[1]
    ax.clear()
    ax.set_axis_off()

    ax_scatter = inset_axes(ax, width="75%", height="75%", loc="lower left", borderpad=0)
    ax_scatter.set_xlabel(f"PC {dir1}")
    ax_scatter.set_ylabel(f"PC {dir2}")

    ax_hist_x = inset_axes(ax, width="75%", height="25%", loc="upper left", borderpad=0)
    ax_hist_y = inset_axes(ax, width="25%", height="75%", loc="lower right", borderpad=0)
    ax_hist_x.set_axis_off()
    ax_hist_y.set_axis_off()

    if data2 is None:
        size_scat = 2
    else:
        size_scat = 50
    ax_scatter.scatter(
        data1[:, dir1],
        data1[:, dir2],
        color="black",
        s=size_scat,
        zorder=0,
        alpha=0.3,
    )
    _, bins_x, _ = ax_hist_x.hist(
        data1[:, dir1],
        bins=40,
        color="black",
        histtype="step",
        zorder=0,
        density=True,
        orientation="vertical",
        lw=1,
        label=label_1,
    )
    _, bins_y, _ = ax_hist_y.hist(
        data1[:, dir2],
        bins=40,
        color="black",
        histtype="step",
        zorder=0,
        density=True,
        orientation="horizontal",
        lw=1,
    )
    if data2 is not None:
        ax_scatter.scatter(
            data2[:, dir1],
            data2[:, dir2],
            color="red",
            s=20,
            zorder=2,
            edgecolor="black",
            marker="o",
            alpha=1,
            linewidth=0.4,
        )
        ax_hist_x.hist(
            data2[:, dir1],
            bins=bins_x,
            color="red",
            histtype="step",
            zorder=0,
            density=True,
            orientation="vertical",
            lw=1,
            label=label_2,
        )
        ax_hist_y.hist(
            data2[:, dir2],
            bins=bins_y,
            color="red",
            histtype="step",
            zorder=0,
            density=True,
            orientation="horizontal",
            lw=1,
        )
    if labels is not None:
        ax_hist_x.legend(fontsize=12, bbox_to_anchor=(1, 1))


def plot_mult_PCA(
    data1: np.ndarray,
    data2: Optional[np.ndarray] = None,
    labels: Optional[List[str]] = None,
    n_dir: int = 2,
):
    if data2 is not None:
        if data2.shape[1] < data1.shape[1]:
            raise ValueError(
                f"data2 should have at least as many components as data1. data1 : {data1.shape[1]} vs data2 : {data2.shape[1]}"
            )
        if labels is not None:
            if len(labels) < 2:
                raise ValueError(
                    f"There should be 2 labels, got {len(labels)} : {labels}"
                )

    max_cols = 4
    n_cols = min(data1.shape[1] // 2, max_cols)
    n_plots = data1.shape[1] // n_dir

    n_rows = (
        (data1.shape[1] // 2) // max_cols
        if n_plots % max_cols == 0
        else ((data1.shape[1] // 2) // max_cols) + 1
    )

    fig, ax = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))

    for i in range(n_rows):
        for j in range(n_cols):
            curr_plot_idx = n_cols * i + j
            indexes = [i, j] if n_rows > 1 else [j]
            if curr_plot_idx < n_plots:
                plot_one_PCA(
                    ax=ax[*indexes],
                    data1=data1,
                    data2=data2,
                    labels=labels if curr_plot_idx == 0 else None,
                    dir1=curr_plot_idx * 2,
                    dir2=curr_plot_idx * 2 + 1,
                )
            else:
                ax[*indexes].set_axis_off()
    plt.subplots_adjust(wspace=0.35)

    return fig, ax


def process_corr(data_1: np.ndarray, data_2: np.ndarray, threshold: float = 0.0):
    data_1 = data_1.flatten()
    data_2 = data_2.flatten()
    mask = np.logical_not(np.isnan(data_1)) & np.logical_not(np.isnan(data_2))
    data_1 = data_1[mask]
    data_2 = data_2[mask]

    mask = (np.abs(data_1) > threshold) | (np.abs(data_2) > threshold)
    data_1 = data_1[mask]
    data_2 = data_2[mask]

    m, _ = np.polyfit(data_1, data_2, 1)
    r = np.corrcoef(data_1, data_2)[0, 1]

    x_line = np.linspace(
        np.min([data_1.min(), data_2.min()]), np.max([data_1.max(), data_2.max()]), 100
    )

    return m, r, data_1, data_2, x_line
