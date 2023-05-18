from confseq.misc import get_ci_seq
import matplotlib.pyplot as plt
from dpconc.mechanisms import nprr, laplace, r_G_opt_entropy
from dpconc.dpcs import (
    laplace_hoeffding_ci,
    nprr_hoeffding_ci,
    nprr_hedged_ci,
    nprr_empbern_ci,
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

n = 10000
t = np.arange(1, n + 1)
alpha = 0.1
eps = 2
times = np.unique(
    np.round(np.logspace(1, np.log(n), num=50, base=np.e))
).astype(int)

laplace_mechanism = PrivacyMechanism(
    name="Laplace", eps=eps, mechanism_fn=lambda x, eps=eps: laplace(x, eps=eps)
)

r, G = r_G_opt_entropy(eps)

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
                name="Lap-H",
                cs_fn=lambda z: get_ci_seq(
                    z,
                    ci_fn=lambda z: laplace_hoeffding_ci(
                        z=z, scale=1 / eps, trunc_scale=0.1, alpha=alpha
                    ),
                    times=times,
                    parallel=True,
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
                name="NPRR-H",
                cs_fn=lambda z: get_ci_seq(
                    z,
                    ci_fn=lambda z: nprr_hoeffding_ci(z, r=r_G1, alpha=alpha),
                    times=times,
                    parallel=True,
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
                name="NPRR-EB",
                cs_fn=lambda z: get_ci_seq(
                    z,
                    ci_fn=lambda z: nprr_empbern_ci(
                        z,
                        r=r,
                        alpha=alpha,
                        truncation=0.5,
                        prior_mean=0.5,
                        prior_variance=1 / 8,
                    ),
                    times=times,
                    parallel=False,
                ),
                plot_color="tab:orange",
                plot_linestyle="-.",
            ),
            ConfseqToPlot(
                name="NPRR-hedged",
                cs_fn=lambda z: get_ci_seq(
                    z,
                    ci_fn=lambda z: nprr_hedged_ci(
                        z,
                        r=r,
                        alpha=alpha,
                        trunc_scale=0.8,
                        theta=1 / 2,
                        breaks=1000,
                        parallel=False,
                    ),
                    times=times,
                    parallel=True,
                ),
                plot_color="tab:green",
                plot_linestyle=":",
            ),
        ],
    ),
]


dgp_dict = {
    # "bounded_binomial_0.5": binomial_dgp(p=1 / 2, n=n, name="Bernoulli(1/2)"),
    # "bounded_binomial_0.1": binomial_dgp(p=1 / 10, n=n, name="Bernoulli(1/10)"),
    # "bounded_beta_1_1.pdf": beta_dgp(1, 1, n=n, name="Uniform[0, 1]"),
    "bounded_beta_50_50": beta_dgp(50, 50, n=n, name="Beta(10, 30)"),
}

save_plotting_data = True
save_figure = True

for (dgp_name, dgp) in dgp_dict.items():
    plot_name = dgp_name + "_ci"
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
        xlabel="Sample size $n$",
    )

    if save_figure:
        plt.savefig("figures/" + plot_name + ".pdf", bbox_inches="tight")

    plt.show()
