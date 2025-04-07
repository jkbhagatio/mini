"""Plotting utility functions."""

from typing import Optional, Union, List

import pandas as pd
import seaborn as sns

from jaxtyping import Float
from matplotlib.axes import Axes
from plotly import graph_objects as go


def plot_l0_stats(l0_history: Float[list[dict], "step l0_mean l0_std alpha"]) -> go.Figure:
    """Creates a plot of L0 std vs mean."""
    l0_fig = go.Figure()
    for point in l0_history:
        l0_fig.add_trace(
            go.Scatter(
                x=[point["mean"]],
                y=[point["std"]],
                mode="markers",
                marker=dict(
                    size=10,
                    color=f"rgba(0, 0, 255, {point['alpha']})"
                ),
                name=f"Step {point['step']}",
                showlegend=False  # Optional: disable legend to avoid clutter
            )
        )
    l0_fig.update_layout(
        title="L0 std vs mean",
        xaxis_title="L0 mean",
        yaxis_title="L0 std"
    )
    return l0_fig


def box_strip_plot(
    ax: Axes,
    data: Union[pd.DataFrame, List[Float]],
    x: Optional[str] = None,
    y: Optional[str] = None,
    hue: Optional[str] = None,
    show_legend: bool = False
) -> Axes:
    """Creates a stylized combined boxplot and stripplot."""
    # Create boxplot
    sns.boxplot(
        data=data,
        x=x,
        y=y,
        hue=hue,
        width=0.4,
        showfliers=False,
        showmeans=True,
        meanprops={"markersize": "7", "markerfacecolor": "white", "markeredgecolor": "white"},
        legend=show_legend,
        ax=ax,
    )
    
    # Create stripplot
    sns.stripplot(
        data=data,
        x=x,
        y=y,
        hue=hue,
        size=2,
        alpha=0.4,
        dodge=True,
        jitter=True,
        legend=False,
        ax=ax,
    )
    ax.grid(True, alpha=0.5)
    
    return ax


def firing_rate_hist(
    spk_cts: pd.DataFrame,  # indexed by time (s)
) -> Axes:
    """Creates a histogram of firing rates."""
    fr = spk_cts.sum() / (spk_cts.index[-1] - spk_cts.index[0])
    ax = sns.histplot(fr)
    ax.set_xlabel("Firing rate (hz)")
    ax.set_ylabel("Unit count")
    ax.set_title(
        f"Distribution of firing rates; n_units = {len(fr)}, tot_n_spikes = {spk_cts.sum().sum()}"
    )
    ax.grid(True, alpha=0.5)
    return ax
