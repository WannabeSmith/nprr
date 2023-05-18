"""
Plotting functions and utilities for confidence sequences and intervals, privacy mechanisms, etc.
"""
from confseq.cs_plots import DataGeneratingProcess
from nprr.dpcs import PrivacyMechanism
from typing import List, NamedTuple, Callable, Tuple, Union
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from scipy.stats import binom, beta
import dill


class ConfseqToPlot(NamedTuple):
    """
    Confidence sequence

    Attributes
    ----------
    name, String
        Name of the confidence sequence
    """

    name: str
    cs_fn: Callable[
        [NDArray[np.float_]], Tuple[NDArray[np.float_], NDArray[np.float_]]
    ]
    plot_color: str
    plot_linestyle: str


class MeanToPlot(NamedTuple):
    """
    Mean to plot

    Attributes
    ----------
    name, String
        Name of the confidence sequence
    """

    mean: NDArray[np.float_]
    label: str
    color: str
    linestyle: str


class PrivacyMechanismC2PListPair(NamedTuple):
    """
    A PrivacyMechanism - ConfseqToPlot pair

    Attributes
    ----------
    name, String
        Name of the pair
    """

    name: str
    privacy_mechanism: PrivacyMechanism
    confseq_list: List[ConfseqToPlot]


def binomial_dgp(p: Union[float, NDArray[np.float_]], n: int, name: str):
    return DataGeneratingProcess(
        name,
        data_generator_fn=lambda: np.random.binomial(1, p, n),
        discrete=True,
        dist_fn=lambda x: binom.pmf(x, 1, p),
        mean=p,
        title=None,
    )


def beta_dgp(a: float, b: float, n: int, name: str):
    return DataGeneratingProcess(
        name,
        data_generator_fn=lambda: np.random.beta(a, b, n),
        discrete=False,
        dist_fn=lambda x: beta.pdf(x, a, b),
        mean=a / (a + b),
        title=None,
    )


def generate_plotting_data(
    dgp: DataGeneratingProcess,
    filename: str,
    pair_list: List[PrivacyMechanismC2PListPair],
    nsim: int,
    save=False,
):
    # Dictionary of lower/upper CSs.
    # Key is CS name, value is lower/upper CS resp.
    lower_dict = {}
    upper_dict = {}
    # Same for colors and linestyles
    colors_dict = {}
    linestyles_dict = {}

    for _ in range(nsim):
        x = dgp.data_generator_fn()
        for priv_mech_cs_pair in pair_list:
            priv_mech = priv_mech_cs_pair.privacy_mechanism
            confseqs = priv_mech_cs_pair.confseq_list
            z = priv_mech.mechanism_fn(x)
            for confseq in confseqs:
                lower, upper = confseq.cs_fn(z)
                # If we've not yet added this cs to the dictionary
                if confseq.name not in lower_dict:
                    lower_dict[confseq.name] = lower / nsim
                    upper_dict[confseq.name] = upper / nsim
                else:  # CS is already in dictionary.
                    # Add lower/upper to the average
                    lower_dict[confseq.name] += lower / nsim
                    upper_dict[confseq.name] += upper / nsim

                colors_dict[confseq.name] = confseq.plot_color
                linestyles_dict[confseq.name] = confseq.plot_linestyle

    # grid = np.arange(0, 1.01, step=0.01)
    # if empirical_dist:
    #     ax_dist.hist(dgp.data_generator_fn())
    # else:
    #     if dgp.discrete:
    #         ax_dist.bar(grid, dgp.dist_fn(grid), width=0.1)
    #         ax_dist.set_ylabel("pmf")
    #     else:
    #         ax_dist.plot(grid, dgp.dist_fn(grid))
    #         ax_dist.set_ylabel("pdf")

    confseq_obj_list = [
        confseq_obj
        for priv_mech_cs_pair in pair_list
        for confseq_obj in priv_mech_cs_pair.confseq_list
    ]

    # To serialize and dump
    plotting_data = {
        "confseq_obj_list": confseq_obj_list,
        "lower_dict": lower_dict,
        "upper_dict": upper_dict,
        "colors_dict": colors_dict,
        "linestyles_dict": linestyles_dict,
    }

    if save:
        with open(filename, "wb") as fp:  # Pickling
            print("Dumping plotting data to " + filename)
            dill.dump(plotting_data, fp)

    return plotting_data


def plot_cs(
    plotting_data,
    figsize: Tuple[Union[float, int], Union[float, int]],
    times=None,
    width: bool = False,
    ylabel="Confidence sequence width",
    xlabel=r"Sample size $t$",
    mean_data: Union[MeanToPlot, None] = None,
    plot_alpha: float = 1,
    start_time=10,
):
    confseq_obj_list = plotting_data["confseq_obj_list"]
    lower_dict = plotting_data["lower_dict"]
    upper_dict = plotting_data["upper_dict"]
    colors_dict = plotting_data["colors_dict"]
    linestyles_dict = plotting_data["linestyles_dict"]

    if times is None:
        confseq_name_0 = confseq_obj_list[0].name
        times = np.arange(1, len(lower_dict[confseq_name_0]) + 1)

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "dejavuserif"
    plt.rcParams["font.size"] = 13
    fig, ax_cs = plt.subplots(1, 1, figsize=figsize)

    if mean_data is not None:
        ax_cs.plot(
            times,
            mean_data.mean,
            label=mean_data.label,
            color=mean_data.color,
            linestyle=mean_data.linestyle,
        )
    for confseq_obj in confseq_obj_list:
        confseq_name = confseq_obj.name
        lower = lower_dict[confseq_name]
        upper = upper_dict[confseq_name]
        color = colors_dict[confseq_name]
        linestyle = linestyles_dict[confseq_name]
        if width:
            if isinstance(linestyle, str):
                ax_cs.plot(
                    times,
                    upper - lower,
                    color=color,
                    linestyle=linestyle,
                    label=confseq_name,
                    alpha=plot_alpha,
                )
            else:
                ax_cs.plot(
                    times,
                    upper - lower,
                    color=color,
                    dashes=linestyle,
                    label=confseq_name,
                    alpha=plot_alpha,
                )

            ax_cs.set_ylabel(ylabel)
        else:
            if isinstance(linestyle, str):
                ax_cs.plot(
                    times,
                    lower,
                    color=color,
                    linestyle=linestyle,
                    label=confseq_name,
                    alpha=plot_alpha,
                )
                ax_cs.plot(
                    times,
                    upper,
                    color=color,
                    linestyle=linestyle,
                    alpha=plot_alpha,
                )
            else:
                ax_cs.plot(
                    times,
                    lower,
                    color=color,
                    dashes=linestyle,
                    label=confseq_name,
                    alpha=plot_alpha,
                )
                ax_cs.plot(
                    times,
                    upper,
                    color=color,
                    dashes=linestyle,
                    alpha=plot_alpha,
                )

            ax_cs.set_ylabel(ylabel)

        ax_cs.set_xlabel(xlabel)
        ax_cs.set_xscale("log")
        ax_cs.set_xlim(left=start_time)

    ax_cs.legend(loc="best")
