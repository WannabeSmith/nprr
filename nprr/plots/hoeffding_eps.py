import matplotlib.pyplot as plt
from dpconc.dpcs import nprr_hoeffding_ci
from dpconc.plotting import (
    ConfseqToPlot,
    PrivacyMechanismC2PListPair,
    beta_dgp,
    generate_plotting_data,
    plot_cs,
)
from dpconc.mechanisms import nprr
from dpconc import PrivacyMechanism
import numpy as np
import math
from confseq.betting import get_ci_seq


n = 100000
dgfn = lambda: np.random.binomial(1, 0.5, n)
t = np.arange(1, n + 1)
alpha = 0.1
eps_list = [0.5, 1, 2, 4, math.inf]
colors = ["tab:blue", "tab:orange", "tab:red", "tab:purple", "tab:green"]
linestyles = ["-.", [1, 2, 1, 2, 1, 6], "--", ":", "-"]

times = np.unique(
    np.round(np.logspace(1, np.log(n), num=50, base=np.e))
).astype(int)

pair_list = [None] * len(eps_list)

for i in range(len(eps_list)):
    eps = eps_list[i]
    if eps == math.inf:
        nprr_mechanism = PrivacyMechanism(
            name="Identity",
            eps=eps,
            mechanism_fn=lambda x: x,
        )

        confseq_list = [
            ConfseqToPlot(
                name=r"$\varepsilon = \infty$",
                cs_fn=lambda z: get_ci_seq(
                    z,
                    ci_fn=lambda z: nprr_hoeffding_ci(
                        z, r=1, alpha=alpha, running_intersection=False
                    ),
                    times=times,
                    parallel=True,
                ),
                plot_color=colors[i],
                plot_linestyle=linestyles[i],
            ),
        ]

        pair_list[i] = PrivacyMechanismC2PListPair(
            name="Identity-Hoeffding",
            privacy_mechanism=nprr_mechanism,
            confseq_list=confseq_list,
        )
    else:
        r, G = (np.exp(eps) - 1) / (np.exp(eps) + 1), 1

        print("r: " + str(r))
        print("G: " + str(G))
        print("epsilon: " + str(np.log(1 + r * (G + 1) / (1 - r))))

        nprr_mechanism = PrivacyMechanism(
            name="NPRR",
            eps=eps,
            mechanism_fn=lambda x, r=r, G=G: nprr(x, r=r, G=G),
        )

        confseq_list = [
            ConfseqToPlot(
                name=r"$\varepsilon = " + str(eps) + "$",
                cs_fn=lambda z, r=r: get_ci_seq(
                    z,
                    ci_fn=lambda z, r=r: nprr_hoeffding_ci(
                        z, r=r, alpha=alpha, running_intersection=False
                    ),
                    times=times,
                    parallel=True,
                ),
                plot_color=colors[i],
                plot_linestyle=linestyles[i],
            ),
        ]

        pair_list[i] = PrivacyMechanismC2PListPair(
            name="NPRR-Hoeffding",
            privacy_mechanism=nprr_mechanism,
            confseq_list=confseq_list,
        )

dgp_dict = {
    # "bounded_binomial_0.5": binomial_dgp(p=1 / 2, n=n, name="Bernoulli(1/2)"),
    "bounded_beta_1_1": beta_dgp(1, 1, n=n, name="Uniform[0, 1]"),
    # "bounded_beta_10_30": beta_dgp(10, 30, n=n, name="Beta(10, 30)"),
}

save_plotting_data = True
save_figure = True

for (dgp_name, dgp) in dgp_dict.items():
    plot_name = dgp_name + "_hoeffding_eps"
    plotting_data = generate_plotting_data(
        dgp=dgp,
        filename="data_dump/" + plot_name + "_serialized",
        pair_list=pair_list,
        nsim=10,
        save=save_plotting_data,
    )
    plot_cs(
        plotting_data,
        figsize=(6, 3.5),
        times=times,
        width=True,
        ylabel="Confidence interval width",
        xlabel=r"Sample size $n$",
    )

    if save_figure:
        plt.savefig("figures/" + plot_name + ".pdf", bbox_inches="tight")

    plt.show()
