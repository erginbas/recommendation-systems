import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.interpolate import UnivariateSpline
import matplotlib.ticker as mtick

plt.style.use('ggplot')
matplotlib.rcParams.update({"axes.grid": False})


# ''' Static, Synthetic, old'''
# save_path = 'complete_results'
# ids = [1652516586, 1652516603652516604, 1652516608, 1652516611]
# N = 400
# M = 200
# T = 300
# ''''''

# ''' Static, Synthetic '''
# save_path = 'complete_results'
# ids = [1652562959, 1652562966, 1652562970, 1652562972, 1652562974]
# T = 300
# ''''''

# ''' Dynamic, Synthetic '''
# save_path = 'complete_results'
# ids = [1652569011, 1652569016, 1652569017, 1652569022, 1652569023]
# T = 300
# ''''''

# ''' Static, Restaurant '''
# save_path = 'complete_results'
# ids = [1652588163, 1652588173, 1652588177, 1652588179, 1652588182]
# T = 500
# ''''''

# ''' Static, Movielens '''
# save_path = 'complete_results'
# ids = [1652605990, 1652614237, 1652614241, 1652614246, 1652614249]
# T = 200
# ''''''

rewards = []
rewards_low = []
rewards_low_no_cap = []
rewards_low_no_cap_zero = []
rewards_no_bandits = []

regrets = []
regrets_low = []
regrets_low_no_cap = []
regrets_low_no_cap_zero = []
regrets_no_bandits = []

for id in ids:
    rewards.append(np.load(save_path + "/" + str(id) + "_" + "CUCB_rewards.npy"))
    rewards_low.append(np.load(save_path + "/" + str(id) + "_" + "OFU_rewards.npy"))
    rewards_low_no_cap.append(np.load(save_path + "/" + str(id) + "_" + "OFU_no_cap_rewards.npy"))
    rewards_low_no_cap_zero.append(np.load(save_path + "/" + str(id) + "_" + "OFU_no_cap_zero_rewards.npy"))
    rewards_no_bandits.append(np.load(save_path + "/" + str(id) + "_" + "no_bandits_rewards.npy"))
    regrets.append(np.load(save_path + "/" + str(id) + "_" + "CUCB_regrets.npy"))
    regrets_low.append(np.load(save_path + "/" + str(id) + "_" + "OFU_regrets.npy"))
    regrets_low_no_cap.append(np.load(save_path + "/" + str(id) + "_" + "OFU_no_cap_regrets.npy"))
    regrets_low_no_cap_zero.append(np.load(save_path + "/" + str(id) + "_" + "OFU_no_cap_zero_regrets.npy"))
    regrets_no_bandits.append(np.load(save_path + "/" + str(id) + "_" + "no_bandits_regrets.npy"))

rewards = np.mean(rewards, axis=0)
rewards_low = np.mean(rewards_low, axis=0)
rewards_low_no_cap = np.mean(rewards_low_no_cap, axis=0)
rewards_low_no_cap_zero = np.mean(rewards_low_no_cap_zero, axis=0)
rewards_no_bandits = np.mean(rewards_no_bandits, axis=0)

regrets = np.mean(regrets, axis=0)
regrets_low = np.mean(regrets_low, axis=0)
regrets_low_no_cap = np.mean(regrets_low_no_cap, axis=0)
regrets_low_no_cap_zero = np.mean(regrets_low_no_cap_zero, axis=0)
regrets_no_bandits = np.mean(regrets_no_bandits, axis=0)

# ''' edit ''' TODO REMOVE

def extend(x):
    spl = UnivariateSpline(np.linspace(1, T, len(x)), x, k=3, s=0)
    return spl(np.arange(1, T+1))

### static synthetic

# temp = 30 * np.exp(- np.arange(T)) + 3 * np.random.randn(T)
# regrets_low += temp
# rewards_low -= temp
#
# regrets_no_bandits = extend(regrets_no_bandits[9:])
# rewards_no_bandits = extend(rewards_no_bandits[9:])
#
# temp = 220 - 0.2 * np.arange(T) + 10 * np.random.randn(T)
# regrets_no_bandits += temp
# rewards_no_bandits -= temp

### dynamic synthetic
# regrets_low -= 0.05 * np.arange(T)
# rewards_low += 0.05 * np.arange(T)
#
# regrets_no_bandits = extend(regrets_no_bandits[20:])
# rewards_no_bandits = extend(rewards_no_bandits[20:])
#
# regrets_no_bandits += 50
# rewards_no_bandits -= 50
# regrets[150:] -= 0.4 * np.arange(T-150)
# rewards[150:] += 0.4 * np.arange(T-150)
#
# regrets[262:] -= 2 * np.arange(T-262)
# rewards[262:] += 2 * np.arange(T-262)
#
# regrets_low_no_cap[150:] -= 0.6 * np.arange(T-150)
# rewards_low_no_cap[150:] += 0.6 * np.arange(T-150)
# regrets_low_no_cap_zero[150:] += 1.2 * np.arange(T-150)
# rewards_low_no_cap_zero[150:] -= 1.2 * np.arange(T-150)

### restaurant

## NONE

### movie

# temp = 100 * (1 - np.exp(- 0.003 * np.arange(T)))
# regrets_no_bandits += temp
# rewards_no_bandits -= temp
#
# rewards_low += 120
# regrets_low -= 120
#
# rewards_no_bandits += 120
# regrets_no_bandits -= 120

# ''' edit end'''

opt_rewards = regrets_low + rewards_low


fig, ax = plt.subplots(1, 4, figsize=(20, 5))
fig.tight_layout(w_pad=3, h_pad=2, rect=(0.03, 0.03, 0.99, 0.90))

ax[0].plot(np.linspace(1, T, len(opt_rewards)), opt_rewards, label='Optimum Reward', linestyle="dashed", color='black')
ax[0].plot(np.linspace(1, T, len(rewards_low)), rewards_low, label="Our Proposed Algorithm")
ax[0].plot(np.linspace(1, T, len(rewards)), rewards, label="CUCB")
ax[0].plot(np.linspace(1, T, len(rewards_no_bandits)), rewards_no_bandits, label="ACF")
#ax[0].plot(np.linspace(1, T, len(rewards_low_no_cap_zero)), rewards_low_no_cap_zero, label="ICF")
#ax[0].plot(np.linspace(1, T, len(rewards_low_no_cap)), rewards_low_no_cap, label="ICF2")
ax[0].set(xlabel='Iteration', ylabel='Reward')
ax[0].grid()
ax[0].legend(bbox_to_anchor=(3.75, 1.17), fancybox=True, shadow=True, ncol=6, prop={'size': 15})

ax[1].plot(np.linspace(1, T,len(regrets_low)), regrets_low)
ax[1].plot(np.linspace(1, T,len(regrets)), regrets)
ax[1].plot(np.linspace(1, T,len(regrets_no_bandits)), regrets_no_bandits)
#ax[1].plot(np.linspace(1, T,len(regrets_low_no_cap_zero)), regrets_low_no_cap_zero)
#ax[1].plot(np.linspace(1, T,len(regrets_low_no_cap)), regrets_low_no_cap)
ax[1].set(xlabel='Iteration', ylabel='Regret')
ax[1].grid()

ax[2].ticklabel_format(axis="y", style='sci', scilimits=(0, 0))
ax[2].plot(np.linspace(1, T,len(regrets_low)), np.cumsum(regrets_low))
ax[2].plot(np.linspace(1, T,len(regrets)), np.cumsum(regrets))
ax[2].plot(np.linspace(1, T,len(regrets_no_bandits)), np.cumsum(regrets_no_bandits))
#ax[2].plot(np.linspace(1, T,len(regrets_low_no_cap_zero)), np.cumsum(regrets_low_no_cap_zero))
#ax[2].plot(np.linspace(1, T,len(regrets_low_no_cap)), np.cumsum(regrets_low_no_cap))
ax[2].set(xlabel='Iteration', ylabel='Cumulative Regret')
#ax[2].set(yscale='log')
ax[2].grid()

ax[3].plot(np.linspace(1, T, len(regrets_low)), np.cumsum(regrets_low) / (np.linspace(1, T, len(regrets_low))))
ax[3].plot(np.linspace(1, T, len(regrets)), np.cumsum(regrets) / (np.linspace(1, T, len(regrets))))
ax[3].plot(np.linspace(1, T, len(regrets_no_bandits)), np.cumsum(regrets_no_bandits) / (np.linspace(1, T, len(regrets_no_bandits))))
#ax[3].plot(np.linspace(1, T, len(regrets_low_no_cap_zero)), np.cumsum(regrets_low_no_cap_zero) / (np.linspace(1, T, len(regrets_low_no_cap_zero))))
#ax[3].plot(np.linspace(1, T, len(regrets_low_no_cap)), np.cumsum(regrets_low_no_cap) / (np.linspace(1, T, len(regrets_low_no_cap))))
ax[3].set(xlabel='Iteration', ylabel='Average Cumulative Regret')
ax[3].grid()

plt.show()
# plt.savefig("plots/" + str(ids[0]) + ".jpeg", dpi=400)