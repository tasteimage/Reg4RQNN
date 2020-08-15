import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
"""
loss
categorical_accuracy
val_loss
val_categorical_accuracy
"""

MODELS = ["l1", "l2", "rq", "none"]
RATE = ["5e-2", "1e-2", "5e-3", "1e-3", "5e-4", "1e-4", "5e-5", "1e-5"]

def get_data(name):
    res = np.zeros((4, 8))

    for i, model in enumerate(MODELS):
        for j, rate in enumerate(RATE):
            # We choose to calculate the mean value of last 10 data 
            # in order to aleviate the data fluctuation
            log = np.load(
                "./mnist_{}/exp-{}/log.npy".format(rate, model),
                allow_pickle=True
            )
            res[i, j] = np.mean(log.item().get(name)[-10:])

    return res


def heatmap(data, title, row_labels, col_labels, vmin, vmax, func, func_inv, cbar_kw={}, cbarlabel="", **kwargs):
    fig, ax = plt.subplots(figsize=(8, 3))
    # plt.title(title)

    # Plot the heatmap
    im = ax.imshow(func(data), vmin = vmin, vmax = vmax, **kwargs)

    # Create colorbar
    tickers = np.linspace(vmin, vmax, 5)
    cbar = ax.figure.colorbar(
            im,
            ax=ax,
            ticks = tickers,
            format = ticker.FixedFormatter(['{0:.3f}'.format(func_inv(x)) for x in tickers]),
            **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(["$\ell_{1}$", "$\ell_{2}$", "$\ell_{rq}$", "none"])
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right", rotation_mode="anchor")

    for _, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            im.axes.text(j, i, '{0:.3f}'.format(data[i, j]), horizontalalignment="center", verticalalignment="center")

    fig.tight_layout()
    # plt.savefig("{}.png".format(title), bbox_inches="tight", pad_inches=0.2, dpi=300)
    # plt.close()
    plt.show()

if __name__ == "__main__":
    heatmap(get_data("loss"), "Loss", MODELS, RATE, func = lambda x: x, func_inv = lambda x: x, vmin=0, vmax=2.5, cmap='Blues')
    heatmap(get_data("categorical_accuracy"), "Accuracy", MODELS, RATE, func = lambda x: x ** 10, func_inv = lambda x: x ** (1 / 10), vmin=0, vmax=1, cmap='Blues')
    heatmap(get_data("val_loss"), "Validation Loss", MODELS, RATE, func = lambda x: x, func_inv = lambda x: x, vmin=0, vmax=2.5, cmap='Blues')
    heatmap(get_data("val_categorical_accuracy"), "Validation Accuracy", MODELS, RATE, func = lambda x: x ** 10, func_inv = lambda x: x ** (1 / 10), vmin=0, vmax=1, cmap='Blues')
