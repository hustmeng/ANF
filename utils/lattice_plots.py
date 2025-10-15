import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon, Circle


def plaquette(x, y, *, d=0.1, l=0.35, r=0.025, color='black', ax=None):
    if ax is None:
        ax = plt.gca()
    xl = x + d
    yd = y + d
    xr = x + 1 - d
    yu = y + 1 - d
    ax.plot([xl, xl, xl + l], [yd + l, yd, yd], color=color)
    ax.plot([xr - l, xr, xr], [yd, yd, yd + l], color=color)
    ax.plot([xr, xr, xr - l], [yu - l, yu, yu], color=color)
    ax.plot([xl + l, xl, xl], [yu, yu, yu - l], color=color)
    ax.add_patch(RegularPolygon((0.5 + x, yd), 3, radius=r, orientation=np.pi / 2, color=color))
    ax.add_patch(RegularPolygon((xl, 0.5 + y), 3, radius=r, orientation=0, color=color))
    ax.add_patch(RegularPolygon((0.5 + x, yu), 3, radius=r, orientation=-np.pi / 2, color=color))
    ax.add_patch(RegularPolygon((xr, 0.5 + y), 3, radius=r, orientation=-np.pi, color=color))
    return ax


def link(x, y, mu, *, color, d=0.05, l=0.3, r=0.05, ax=None):
    if ax is None:
        ax = plt.gca()
    if mu == 1:
        ax.plot([x, x], [y + d, y + d + l], color=color)
        ax.add_patch(RegularPolygon((x, 0.5 + y), 3, radius=r, orientation=0, color=color))
        ax.plot([x, x], [y + 1 - d - l, y + 1 - d], color=color)
    else:
        ax.plot([x + d, x + d + l], [y, y], color=color)
        ax.add_patch(RegularPolygon((x + 0.5, y), 3, radius=r, orientation=-np.pi / 2, color=color))
        ax.plot([x + 1 - d - l, x + 1 - d], [y, y], color=color)

    return ax


def plot_lattice(L, *, color='black', d=0.05, l=0.3, r=0.05, cr=0.075, ax=None):
    if ax is None:
        ax = plt.gca()

    ax.set_axis_off()
    ax.set_xlim(-0.1, L + 0.1)
    ax.set_ylim(-0.1, L + 0.1)
    ax.set_aspect(1.0)
    for x in range(L):
        for y in range(L):
            link(x, y, 0, color=color, d=d, l=l, r=r, ax=ax)
            link(x, y, 1, color=color, d=d, l=l, r=r, ax=ax)
            ax.add_patch(Circle((x, y), radius=cr, facecolor=color))
    return ax


def plot_mask(mask, *, r=0.025, ax=None):
    if ax is None:
        ax = plt.gca()

    plaq_mask = mask[1][0]
    link_mask = mask[0]

    ax.set_axis_off()
    ax.set_xlim(-0.1, 8.1)
    ax.set_ylim(-0.1, 8.1)
    ax.set_aspect(1.0)
    for i in range(8):
        for j in range(8):
            color = 'black'
            if plaq_mask['frozen'][i, j] == 1:
                color = 'green'
            if plaq_mask['passive'][i, j] == 1:
                color = 'magenta'
            if plaq_mask['active'][i, j] == 1:
                color = 'orange'
            plaquette(i, j, color=color, l=0.3, r=r, ax=ax)

            for mu in (0, 1):
                if link_mask[mu, i, j] == 1:
                    link(i, j, mu, color='blue', ax=ax)
    return ax
