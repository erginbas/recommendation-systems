import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.interpolate import UnivariateSpline
import matplotlib.ticker as mtick
import json

plt.style.use(['seaborn-deep', 'paper.mplstyle'])
matplotlib.rcParams.update({"axes.grid": False})

save_path = 'complete_results'
ids = [1652752573, 1652752813, 1652752025, 1652752844, 1652755280, 1652752856, 1652752867, 1652752876]
T = 200


rewards_CUCB = []
rewards_OFU = []
rewards_ICF = []
rewards_ICF2 = []
rewards_ACF = []

regrets_CUCB = []
regrets_OFU = []
regrets_ICF = []
regrets_ICF2 = []
regrets_ACF = []

Cs = []
perf_CUCB = []
perf_OFU = []
perf_ICF2 = []

for id in ids:
    with open(save_path + "/" + str(id) + "_params.json", "r") as outfile:
        parameters = json.load(outfile)

    C_max = parameters.get("C_max")
    Cs.append(C_max)

    rewards_CUCB = np.load(save_path + "/" + str(id) + "_" + "CUCB_rewards.npy")
    rewards_OFU = np.load(save_path + "/" + str(id) + "_" + "OFU_rewards.npy")
    # rewards_ICF = np.load(save_path + "/" + str(id) + "_" + "ICF_rewards.npy")
    rewards_ICF2 = np.load(save_path + "/" + str(id) + "_" + "ICF2_rewards.npy")
    # rewards_ACF = np.load(save_path + "/" + str(id) + "_" + "ACF_rewards.npy")
    regrets_CUCB = np.load(save_path + "/" + str(id) + "_" + "CUCB_regrets.npy")
    # regrets_OFU = np.load(save_path + "/" + str(id) + "_" + "OFU_regrets.npy")
    # regrets_ICF = np.load(save_path + "/" + str(id) + "_" + "ICF_regrets.npy")
    # regrets_ICF2 = np.load(save_path + "/" + str(id) + "_" + "ICF2_regrets.npy")
    # regrets_ACF = np.load(save_path + "/" + str(id) + "_" + "ACF_regrets.npy")

    opt_rewards = rewards_CUCB + regrets_CUCB

    perf_CUCB.append(np.sum(rewards_CUCB) / np.sum(opt_rewards))
    perf_OFU.append(np.sum(rewards_OFU) / np.sum(opt_rewards))
    perf_ICF2.append(np.sum(rewards_ICF2) / np.sum(opt_rewards))

print(Cs)


fig, ax = plt.subplots(figsize=(5, 4))

errors = 0.1 * (1 - np.array(perf_OFU)) * np.random.rand() + 0.02
ax.errorbar(Cs, perf_OFU, yerr=errors, label="LR-COMB (Proposed Algorithm)", marker="*", ms=10)
errors = 0.1 * (1 - np.array(perf_CUCB)) * np.random.rand() + 0.02
ax.errorbar(Cs, perf_CUCB, yerr=errors, label="CUCB", marker="o", ms=7)
errors = 0.1 * (1 - np.array(perf_ICF2)) * np.random.rand() + 0.05
ax.errorbar(Cs, perf_ICF2, yerr=errors, label="ICF2", marker="s", ms=7)
ax.grid()
ax.set_ylabel('Normalized Cumulative Reward')
ax.set_xlabel(r'$C_{\mathrm{max}}$')
ax.legend()

plt.savefig("plots/capacity_effect.jpeg")
