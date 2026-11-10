"""Shared figure style: recessive axes, validated categorical palette."""
import matplotlib.pyplot as plt

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#8f5cd6"]   # blue, orange, aqua, violet
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"


def setup():
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#c3c2b7",
                         "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "pdf.fonttype": 42})


def style_axes(a):
    a.grid(color=GRID, lw=0.6)
    a.set_axisbelow(True)
    a.set_facecolor("#fcfcfb")
    for side in ("top", "right"):
        a.spines[side].set_visible(False)
