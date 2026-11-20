

from pathlib import Path
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

# =========================================================

# =========================================================



FONT_FAMILY = "serif"
FONT_SERIF = ["Times New Roman", "serif"]
BASE_FONTSIZE = 10
PANEL_CAPTION_FONTSIZE = 10
AX_LABELSIZE = PANEL_CAPTION_FONTSIZE
MULTI_AXIS_LABELSIZE = 9
TITLE_FONTSIZE = 9
TICK_LABELSIZE = 8
AXES_LINEWIDTH = 0.8
GRID_COLOR = "#d9d9d9"
GRID_LINEWIDTH = 0.4
SPINE_COLOR = "#333333"
DPI_FIG = 300
DPI_SAVE = 300
COST_LOSS_XLABEL = r"Defense investment $C(d_{0})$ (\$’000)"
COST_LOSS_YLABEL = r"Expected defender loss $L(d_{0})$ (\$’000)"
_PANEL_CAPTION = re.compile(r"^\s*\([a-z]+\)(?:\s|$)")


def cost_loss_axis_limits(cost_series, loss_series, budget=None, *,
                          x_start=-1, y_start=-1, x_end=None, y_end=None,
                          margin=0.05, min_xspan=1.0, min_yspan=1.0):
    """Resolve comparable monetary axes from exported costs, losses and budget."""
    def upper(series, start, end, minimum_span):
        if end is not None:
            return float(end)
        maxima = []
        for values in series:
            values = np.asarray(values, dtype=float).reshape(-1)
            finite = values[np.isfinite(values)]
            if finite.size:
                maxima.append(float(finite.max()))
        if not maxima:
            return float(start) + max(minimum_span, 1.0)
        span = max(maxima) - float(start)
        return float(start) + (span * (1 + margin) if span >= minimum_span else minimum_span)

    costs = list(cost_series)
    if budget is not None:
        costs.append([budget])
    return (
        (float(x_start), upper(costs, x_start, x_end, min_xspan)),
        (float(y_start), upper(loss_series, y_start, y_end, min_yspan)),
    )


def format_cost_loss_axes(axis, limits):
    """Apply identical labels, monetary units, limits and tick policies."""
    axis.set_xlabel(COST_LOSS_XLABEL)
    axis.set_ylabel(COST_LOSS_YLABEL)
    axis.set_xlim(*limits[0])
    axis.set_ylim(*limits[1])
    axis.xaxis.set_major_locator(MaxNLocator(nbins=7, steps=[1, 2, 5, 10]))
    axis.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 5, 10]))
    axis.xaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    axis.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))


def plot_style(*, kind="single", legend_fontsize=None):
    
    if kind not in ("single", "multi", "auto"):
        raise ValueError("kind must be single, multi or auto")
    settings = {
        "font.family": FONT_FAMILY, "font.serif": FONT_SERIF,
        "font.size": BASE_FONTSIZE,
        "axes.labelsize": MULTI_AXIS_LABELSIZE if kind == "multi" else AX_LABELSIZE,
        "axes.titlesize": TITLE_FONTSIZE, "figure.titlesize": TITLE_FONTSIZE,
        "xtick.labelsize": TICK_LABELSIZE, "ytick.labelsize": TICK_LABELSIZE,
        "axes.linewidth": AXES_LINEWIDTH, "xtick.major.width": AXES_LINEWIDTH,
        "ytick.major.width": AXES_LINEWIDTH, "lines.linewidth": 1.5,
        "figure.dpi": DPI_FIG, "savefig.dpi": DPI_SAVE, "savefig.bbox": "tight",
        "axes.unicode_minus": False, "mathtext.fontset": "stix",
        "svg.fonttype": "path",
    }
    if legend_fontsize is not None:
        settings["legend.fontsize"] = legend_fontsize
    return settings


def set_plot_style(*, kind="single", legend_fontsize=None):
    
    mpl.rcParams.update(plot_style(kind=kind, legend_fontsize=legend_fontsize))


def _is_multi_panel(fig):
    
    positions = {
        tuple(round(value, 5) for value in ax.get_position().bounds)
        for ax in fig.axes
        if ax.get_label() != "<colorbar>" and ax.axison and ax.get_visible()
    }
    return len(positions) > 1


def normalize_figure_typography(fig, *, kind="auto"):
    
    if kind not in ("single", "multi", "auto"):
        raise ValueError("kind must be single, multi or auto")
    multi = _is_multi_panel(fig) if kind == "auto" else kind == "multi"
    label_size = MULTI_AXIS_LABELSIZE if multi else AX_LABELSIZE

    def apply(text, size):
        text.set_fontfamily(FONT_SERIF[0])
        text.set_fontsize(size)

    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            apply(axis.label, label_size)
            apply(axis.get_offset_text(), TICK_LABELSIZE)
            axis.set_tick_params(which="both", labelsize=TICK_LABELSIZE)
            for tick in (*axis.get_major_ticks(), *axis.get_minor_ticks()):
                apply(tick.label1, TICK_LABELSIZE)
                apply(tick.label2, TICK_LABELSIZE)
        for title in (ax.title, ax._left_title, ax._right_title):
            apply(title, PANEL_CAPTION_FONTSIZE if _PANEL_CAPTION.match(title.get_text())
                  else TITLE_FONTSIZE)
        for text in ax.texts:
            apply(text, PANEL_CAPTION_FONTSIZE if _PANEL_CAPTION.match(text.get_text())
                  else TICK_LABELSIZE)
    for text in fig.texts:
        if text in (fig._supxlabel, fig._supylabel):
            size = label_size
        elif _PANEL_CAPTION.match(text.get_text()):
            size = PANEL_CAPTION_FONTSIZE
        else:
            size = TITLE_FONTSIZE
        apply(text, size)
    return fig


def style_axes(ax):
    
    ax.grid(True, axis="y", linestyle="--", linewidth=GRID_LINEWIDTH,
            color=GRID_COLOR, zorder=0)
    for spine in ax.spines.values():
        spine.set_color(SPINE_COLOR)
        spine.set_linewidth(AXES_LINEWIDTH)


def save_figure(fig, directory, basename):
    
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    normalize_figure_typography(fig)
    for extension in ("png", "svg"):
        fig.savefig(directory / f"{basename}.{extension}", format=extension,
                    bbox_inches="tight")
    plt.close(fig)
