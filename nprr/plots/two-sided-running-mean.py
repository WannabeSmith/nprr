import matplotlib.pyplot as plt
import math
from dpconc.mechanisms import nprr
from dpconc.dpcs import (
    nprr_twosided_runningmean_cs,
    nprr_onesided_runningmean_cs,
)
from dpconc import PrivacyMechanism
from dpconc.plotting import (
    plot_cs,
    generate_plotting_data,
    binomial_dgp,
    ConfseqToPlot,
    MeanToPlot,
    PrivacyMechanismC2PListPair,
)
import numpy as np

n = 100000
t = np.arange(1, n + 1)
alpha = 0.1
eps = 2

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
            ConfseqToPlot(
                name=r"$\widetilde C_t^\pm$",
                cs_fn=lambda z: nprr_twosided_runningmean_cs(
                    z,
                    r=r_G1,
                    t_opt=100,
                    alpha=alpha,
                ),
                plot_color="mediumseagreen",
                plot_linestyle="-",
            ),
            ConfseqToPlot(
                name=r"$\widetilde L_t$",
                cs_fn=lambda z: (
                    nprr_onesided_runningmean_cs(
                        z,
                        r=r_G1,
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
individual_means = (
    1 - np.sin(2 * np.log(np.e + t)) / np.log(np.e + t / 100)
) / 2
running_means = np.cumsum(individual_means) / t

dgp_dict = {
    # "bounded_binomial_0.5": binomial_dgp(p=1 / 2, n=n, name="Bernoulli(1/2)"),
    # "bounded_beta_1_1.pdf": beta_dgp(1, 1, n=n, name="Uniform[0, 1]"),
    "wavy": binomial_dgp(p=individual_means, n=n, name=r"Bernoulli($\mu_t$)"),
}

save_plotting_data = True
save_figure = True

for (dgp_name, dgp) in dgp_dict.items():
    plot_name = dgp_name + "_cs"
    plotting_data = generate_plotting_data(
        dgp=dgp,
        filename="data_dump/" + plot_name + "_serialized",
        pair_list=pair_list,
        nsim=1,
        save=save_plotting_data,
    )
    mean_data = MeanToPlot(
        mean=running_means,
        label=r"$\widetilde \mu_t$",
        color="grey",
        linestyle=":",
    )
    plot_cs(
        plotting_data,
        figsize=(6, 3.5),
        width=False,
        xlabel=r"Time $t$",
        ylabel=r"Confidence sequence",
        mean_data=mean_data,
        plot_alpha = 0.8
    )

    if save_figure:
        plt.savefig("figures/" + plot_name + ".pdf", bbox_inches="tight")

    plt.show()
