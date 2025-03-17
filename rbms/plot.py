import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec


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


def plot_hist(
    ax, data_proj, gen_data_proj, color, proj, labels, orientation="vertical"
):
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
        ] = sample[id_s].reshape(
            shape
        )  # Directly reshape to `shape`

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
