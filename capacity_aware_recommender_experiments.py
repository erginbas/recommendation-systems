# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
import json
from pathlib import Path
import uuid
from multiprocessing import Pool
import itertools
import logging

from data_utils import get_dataset, get_capacity
from Algorithms import Algorithms

plt.style.use('ggplot')
matplotlib.rcParams.update({"axes.grid": False})

"""choose data set"""
dataset = "synthetic"
save_path = 'results'

"""set if dynamic/static"""
is_dynamic = False

""" show plots """
show_plots = False

""" number of monte carlo simulations """
num_sims = 50

"""verbose"""
verbose = True

def run_exp(input):
    data_arg = {}
    if dataset == "synthetic":
        T = 500
        data_arg["N"] = 50
        data_arg["M"] = 20
        data_arg["rank"] = 3
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
    dem_to_cap_ratio = 0.9
    C, D = get_capacity(N, M, T, is_dynamic, dem_to_cap_ratio=dem_to_cap_ratio, p_activity=p_activity)

    # save the simulation configutarion
    exp_params = {"N": N, "M": M, "T": T, "dataset": dataset, "rank": rank,
                  "C_max": int(np.max(C)), "p_activity": p_activity, "dynamic": int(is_dynamic)}

    exp_save_path = f"{save_path}/{input[0]}/{input[1]}"
    Path(exp_save_path).mkdir(exist_ok=True, parents=True)
    with open(f"{exp_save_path}/params.json", "w") as outfile:
        json.dump(exp_params, outfile)

    logging.basicConfig(filename=f"{exp_save_path}/log.out",
                        filemode='a',
                        format='[%(asctime)s] %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.INFO)

    # print summary of parameters
    logging.info(f"N = {N},  M = {M}")
    logging.info(f"total C = {np.sum(C[0])}")
    logging.info(f"total D = {np.sum(D[0])}")

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
            plt.plot(np.arange(T), regrets_list[i][:, 0], label=alg)
        plt.ylabel('Regret')
        plt.xlabel('Iteration')
        plt.grid()
        plt.legend()
        plt.show()


if __name__ == '__main__':
    # set a unique ID for the experiment
    exp_id = uuid.uuid4().hex[:6]
    print(f"ID: {exp_id}")

    query_inputs = list(zip(itertools.repeat(exp_id), range(num_sims)))

    with Pool() as pool:
        results = list(tqdm(pool.imap(run_exp, query_inputs), total=num_sims))

