# This is modified code from https://arxiv.org/abs/2101.08176 by M.S. Albergo et all.


from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt


def init_live_plot(N_era, N_epoch, dpi=125, figsize=(8, 4), plot_ess=True, window=15):
    fig, ax_ess = plt.subplots(dpi=dpi, figsize=figsize)
    ax_ess.set_xlim(window, N_era * N_epoch)
    ax_ess.set_ylim(0, 1)
    ax_ess.grid(plot_ess)
    if plot_ess:
        ess_line = ax_ess.plot([0], [0], alpha=0.5, c='blue', label='ESS')  # dummy
        ax_ess.set_ylabel("ESS")
    else:
        ess_line = []

    ax_loss = ax_ess.twinx()
    ax_loss.grid(False)
    loss_line = ax_loss.plot([0], [0], alpha=0.5, c="orange", label='Loss')  # dummy
    ax_loss.set_ylabel("Loss")

    artists = ess_line + loss_line
    labs = [l.get_label() for l in artists]
    ax_ess.legend(artists, labs, loc=1)
    ax_ess.set_xlabel("Epoch")
    plt.close()

    display_id = display(fig, display_id=True)

    return dict(
        fig=fig,
        ax_ess=ax_ess,
        ax_loss=ax_loss,
        ess_line=ess_line,
        loss_line=loss_line,
        display_id=display_id,
        plot_ess=plot_ess,
        window=window

    )


def moving_average(x, window=10):
    if len(x) < window:
        return np.mean(x, keepdims=True)
    else:
        return np.convolve(x, np.ones(window), "valid") / window


def update_plots(history, fig, ax_ess, ax_loss, ess_line, loss_line, display_id, plot_ess, window):
    if plot_ess:
        Y = np.array(history["ess"])
        Y = moving_average(Y, window=window)
        ess_line[0].set_ydata(Y)
        ess_line[0].set_xdata(np.arange(len(Y)) + window)
    Y = history["loss"]
    Y = moving_average(Y, window=window)
    loss_line[0].set_ydata(np.array(Y))
    loss_line[0].set_xdata(np.arange(len(Y)) + window)
    ax_loss.relim()
    ax_loss.autoscale_view()
    fig.canvas.draw()
    display_id.update(fig)
