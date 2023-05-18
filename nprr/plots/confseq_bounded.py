import matplotlib.pyplot as plt
from dpconc.mechanisms import nprr, laplace, r_G_opt_entropy
from dpconc.dpcs import (
    laplace_hoeffding_cs,
    nprr_empbern_cs,
    nprr_gridKelly_cs,
    nprr_hoeffding_cs,
)
from dpconc import PrivacyMechanism
from dpconc.plotting import (
    plot_cs,
    generate_plotting_data,
    beta_dgp,
    ConfseqToPlot,
    PrivacyMechanismC2PListPair,
)
import numpy as np

n = 100000
t = np.arange(1, n + 1)
alpha = 0.1
eps = 2

laplace_mechanism = PrivacyMechanism(
    name="Laplace", eps=eps, mechanism_fn=lambda x, eps=eps: laplace(x, eps=eps)
)

r, G = r_G_opt_entropy(eps)
print("eps: " + str(np.log(1 + (G + 1) * r / (1 - r))))

print("r: " + str(r))
print("G: " + str(G))

nprr_mechanism = PrivacyMechanism(
    name="NPRR",
    eps=eps,
    mechanism_fn=lambda x, r=r, G=G: nprr(x, r=r, G=G),
)

r_G1 = (np.exp(eps) - 1) / (np.exp(eps) + 1)
nprr_mechanism_G1 = PrivacyMechanism(
    name="NPRR-G=1",
    eps=eps,
    mechanism_fn=lambda x, r=r_G1, G=1: nprr(x, r=r_G1, G=G),
)

pair_list = [
    PrivacyMechanismC2PListPair(
        name="Laplace-Hoeffding",
        privacy_mechanism=laplace_mechanism,
        confseq_list=[
            ConfseqToPlot(
                name="Lap-H-CS",
                cs_fn=lambda z: laplace_hoeffding_cs(
                    z=z, scale=1 / eps, trunc_scale=0.1, alpha=alpha
                ),
                plot_color="tab:red",
                plot_linestyle="-",
            )
        ],
    ),
    PrivacyMechanismC2PListPair(
        name="Hoeffding-NPRR-G=1",
        privacy_mechanism=nprr_mechanism_G1,
        confseq_list=[
            ConfseqToPlot(
                name="NPRR-H-CS",
                cs_fn=lambda z: nprr_hoeffding_cs(
                    z,
                    r=r_G1,
                    alpha=alpha,
                ),
                plot_color="tab:blue",
                plot_linestyle="--",
            ),
        ],
    ),
    PrivacyMechanismC2PListPair(
        name="NPRR-Betting",
        privacy_mechanism=nprr_mechanism,
        confseq_list=[
            ConfseqToPlot(
                name="NPRR-EB-CS",
                cs_fn=lambda z: nprr_empbern_cs(
                    z,
                    r=r,
                    alpha=alpha,
                ),
                plot_color="tab:orange",
                plot_linestyle="-.",
            ),
            ConfseqToPlot(
                name="NPRR-GK-CS",
                cs_fn=lambda z: nprr_gridKelly_cs(
                    z, r=r, D=30, alpha=alpha, parallel=True
                ),
                plot_color="tab:green",
                plot_linestyle=":",
            ),
        ],
    ),
]


dgp_dict = {
    # "bounded_binomial_0.5": binomial_dgp(p=1 / 2, n=n, name="Bernoulli(1/2)"),
    # "bounded_beta_1_1.pdf": beta_dgp(1, 1, n=n, name="Uniform[0, 1]"),
    "bounded_beta_50_50": beta_dgp(50, 50, n=n, name="Beta(50, 50)"),
}

save_plotting_data = True
save_figure = True

for (dgp_name, dgp) in dgp_dict.items():
    plot_name = dgp_name + "_cs"
    plotting_data = generate_plotting_data(
        dgp=dgp,
        filename="data_dump/" + plot_name + "_serialized",
        pair_list=pair_list,
        nsim=10,
        save=save_plotting_data,
    )
    plot_cs(plotting_data, figsize=(6, 3.5), width=True, xlabel=r"Time $t$")

    if save_figure:
        plt.savefig("figures/" + plot_name + ".pdf", bbox_inches="tight")

    plt.show()
