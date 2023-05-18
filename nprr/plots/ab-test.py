from confseq.cs_plots import DataGeneratingProcess
import matplotlib.pyplot as plt
import math

from nprr.mechanisms import nprr
from nprr.dpcs import (
    ipw,
    ipw_to_unit_interval,
    nprr_abtest_lower_cs,
)
from nprr import PrivacyMechanism
from nprr.plotting import (
    plot_cs,
    generate_plotting_data,
    ConfseqToPlot,
    MeanToPlot,
    PrivacyMechanismC2PListPair,
)
import numpy as np

np.random.seed(2022)

n = 10000
t = np.arange(1, n + 1)
alpha = 0.1
eps = 2
pi = 0.5
individual_means_A = 1.8 * (np.exp(t / 300) / (1 + np.exp(t / 300)) - 0.5)
individual_means_B = np.repeat(0.4, len(t))
running_mean_Delta = np.cumsum(individual_means_A - individual_means_B) / t

r_G1 = (np.exp(eps) - 1) / (np.exp(eps) + 1)
nprr_mechanism_G1 = PrivacyMechanism(
    name="NPRR-G=1",
    eps=eps,
    mechanism_fn=lambda x, r=r_G1, G=1: nprr(x, r=r_G1, G=G),
)

pair_list = [
    PrivacyMechanismC2PListPair(
        name="Hoeffding-NPRR-G=1",
        privacy_mechanism=nprr_mechanism_G1,
        confseq_list=[
            # ConfseqToPlot(
            #     name=r"$\widetilde C_t^\pm$",
            #     cs_fn=lambda z: nprr_ab_twosided_cs(
            #         z,
            #         r=r_G1,
            #         t_opt=100,
            #         pi=pi,
            #         alpha=alpha,
            #     ),
            #     plot_color="mediumseagreen",
            #     plot_linestyle="-",
            # ),
            ConfseqToPlot(
                name=r"$\widetilde L_t^\Delta$",
                cs_fn=lambda z: (
                    nprr_abtest_lower_cs(
                        z,
                        r=r_G1,
                        pi=pi,
                        t_opt=100,
                        alpha=alpha,
                    ),
                    np.repeat(math.inf, len(z)),
                ),
                plot_color="tomato",
                plot_linestyle="-",
            ),
        ],
    ),
]


def wavy_ipw_data_generation():
    x_A = np.random.binomial(1, individual_means_A, len(individual_means_A))
    x_B = np.random.binomial(1, individual_means_B, len(individual_means_B))
    treatment = np.random.binomial(1, pi, len(t))

    x = np.where(treatment == 1, x_A, x_B)

    ipw_obs = ipw(obs=x, treatment=treatment, pi=pi)
    return ipw_to_unit_interval(ipw_obs=ipw_obs, pi=pi)


dgp_dict = {
    # "bounded_binomial_0.5": binomial_dgp(p=1 / 2, n=n, name="Bernoulli(1/2)"),
    # "bounded_beta_1_1.pdf": beta_dgp(1, 1, n=n, name="Uniform[0, 1]"),
    "wavy_ipw": DataGeneratingProcess(
        name="Wavy IPW", data_generator_fn=wavy_ipw_data_generation
    )
}

save_plotting_data = True
save_figure = True

for dgp_name, dgp in dgp_dict.items():
    plot_name = dgp_name + "_cs"
    plotting_data = generate_plotting_data(
        dgp=dgp,
        filename="data_dump/" + plot_name + "_serialized",
        pair_list=pair_list,
        nsim=1,
        save=save_plotting_data,
    )
    mean_data = MeanToPlot(
        mean=running_mean_Delta,
        label=r"$\widetilde \Delta_t$",
        color="grey",
        linestyle=":",
    )
    plot_cs(
        plotting_data,
        figsize=(6, 3.5),
        width=False,
        xlabel=r"Time $t$",
        ylabel=r"",
        mean_data=mean_data,
        plot_alpha=0.8,
        start_time=100,
    )

    lower_cs = plotting_data["lower_dict"]["$\widetilde L_t^\Delta$"]

    mean_cross_zero = np.where(running_mean_Delta >= 0)[0][0]
    cs_cross_zero = np.where(lower_cs >= 0)[0][0]

    # Where the mean hits zero
    plt.vlines(
        mean_cross_zero,
        ymin=-3,
        ymax=0,
        color="grey",
        linestyle="-.",
        alpha=0.5,
    )
    plt.text(mean_cross_zero + 50, -1, mean_cross_zero, color="grey")

    # Where the lower cs hits zero
    plt.vlines(
        cs_cross_zero,
        ymin=-3,
        ymax=0,
        color="tomato",
        linestyle="-.",
        alpha=0.5,
    )
    plt.text(cs_cross_zero + 50, -1, cs_cross_zero, color="tomato")

    plt.hlines(
        0,
        xmin=0,
        xmax=np.where(lower_cs >= 0)[0][0],
        color="royalblue",
        linestyle="-.",
        alpha=0.8,
    )

    plt.text(
        95,
        0.08,
        r"$\widetilde \mathcal{H}_0$: $\forall t,\ \widetilde \Delta_t \leq 0$",
        color="royalblue",
        alpha=0.8,
    )

    ### xticks
    # ticks, labels = plt.xticks()
    # print(ticks)
    # ticks = np.append(ticks, 500)
    # labels = np.append(labels, "500")
    # print(labels)
    # # ticks_list = ticks.tolist()
    # # # labels_list = labels.tolist()

    # # ticks_zip = sorted(zip(ticks_list, labels_list))
    # # sorted_labels_list = [element for _, element in ticks_zip]
    # # print(ticks_list)
    # # print(labels_list)
    # # xtl[-1] = "Here is 1.5"
    # # ax.set_xticks(ticks_list)
    # plt.xticks(ticks)
    # ticks,labels=plt.xticks()
    # print(labels)

    plt.legend(loc="lower right")

    plt.ylim(bottom=-1.5)
    plt.xlim(75, 1.2 * 10**4)

    if save_figure:
        plt.savefig("figures/" + plot_name + ".pdf", bbox_inches="tight")

    plt.show()
