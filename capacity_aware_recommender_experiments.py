# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import time
import json
from pathlib import Path
import uuid

from data_utils import get_dataset, get_capacity

from Algorithms import Algorithms

plt.style.use('ggplot')
matplotlib.rcParams.update({"axes.grid": False})

"""### choose data set"""
dataset = "synthetic"
save_path = 'results'

"""### set if dynamic/static"""
is_dynamic = False

""" show plots """
show_plots = True

data_arg = {}

if dataset == "synthetic":
    T = 500
    data_arg["N"] = 50
    data_arg["M"] = 100
    data_arg["rank"] = 10
elif dataset == "restaurant":
    T = 100
elif dataset == "movie":
    T = 100
else:
    raise NotImplementedError

R_true, rank = get_dataset(dataset, **data_arg)

N, M = R_true.shape

# noise sub-gaussianity parameter
eta = 0.2

p_activity = 0.2
C, D = get_capacity(N, M, T, is_dynamic, p_activity=p_activity)

# print summary of parameters
print("N = ", N, ", M = ", M)
print("total C = ", np.sum(C[0]))
print("total D = ", np.sum(D[0]))

# set a unique ID for the experiment
exp_id = uuid.uuid4().hex[:6]
print(exp_id)

# save the simulation configutarion
exp_params = {"N": N, "M": M, "T": T, "dataset": dataset, "rank": rank,
              "C_max": int(np.max(C)), "p_activity": p_activity, "dynamic": int(is_dynamic)}

exp_save_path = f"{save_path}/{str(exp_id)}"
Path(exp_save_path).mkdir(exist_ok=True)
with open(f"{exp_save_path}/params.json", "w") as outfile:
    json.dump(exp_params, outfile)


"""
OFU: allocations with low-rank collaborative filtering (our proposed algorithm)
CUCB: Combinatorial UCB for CMAB without structure)
ACF: Low-rank collaborative filtering without exploration (best allocation w.r.t. LS estimate)
ICF: Interactive collaborative filtering (without capacity constraints)
ICF2: Interactive collaborative filtering (without capacity constraints, 
but observes a zero reward when item is not allocated)
"""

alg_helper = Algorithms(R_true, rank, eta, is_dynamic, C, D, T, exp_save_path)

alg_list = ["OFU", "CUCB", "ACF", "ICF", "ICF2"]
regrets_list = []

for alg in alg_list:
    regrets_list.append(alg_helper.solve_algo(alg))

''' Plots '''
if show_plots:
    for i, alg in enumerate(alg_list):
        print(regrets_list[i])
        plt.plot(np.arange(T), regrets_list[i], label=alg)
    plt.ylabel('Regret')
    plt.xlabel('Iteration')
    plt.grid()
    plt.legend()
    plt.show()

    for i, alg in enumerate(alg_list):
        plt.plot(np.arange(T), np.cumsum(regrets_list[i]), label=alg)
    plt.ylabel('Cumulative Regret')
    plt.xlabel('Iteration')
    plt.grid()
    plt.legend()
    plt.show()

    for i, alg in enumerate(alg_list):
        plt.plot(np.arange(T), np.cumsum(regrets_list[i]) / (1 + np.arange(T)), label=alg)
    plt.ylabel('Average Regret')
    plt.xlabel('Iteration')
    plt.grid()
    plt.legend()
    plt.show()
