import numpy as np
import matplotlib.pyplot as plt


def get_synthetic_dataset(**kwargs):
    N = kwargs["N"]
    M = kwargs["M"]

    # rank of R
    rank = kwargs["rank"]

    # generate low-rank R
    R_temp = np.random.binomial(1, 0.5, size=(N, M))
    u, s, vt = np.linalg.svd(R_temp, full_matrices=False)
    s[rank:] = 0

    R_true = u @ np.diag(s) @ vt
    lower = np.quantile(R_true, 0.999)
    upper = np.quantile(R_true, 0.001)
    R_true = np.clip(R_true, upper, lower)
    R_true = (R_true - lower) / (upper - lower)

    if kwargs.get("plot_data", False):
        plt.hist(R_true.flatten(), bins=np.int32(np.sqrt(N * M)))
        _, s, _ = np.linalg.svd(R_true)
        plt.yscale("log")
        plt.plot(s)
        plt.show()

    return R_true, rank

def get_restaurant_dataset(**kwargs):
    R_true = np.load("data/rating_restaurant_completed.npy")
    # normalize maximum value of R
    R_true = R_true / 6
    # rank of R
    rank = 10

    N, M = R_true.shape

    if kwargs.get("plot_data", False):
        plt.hist(R_true.flatten(), bins=np.int32(np.sqrt(N * M)))
        _, s, _ = np.linalg.svd(R_true)
        plt.yscale("log")
        plt.plot(s)
        plt.show()

    return R_true, rank


def get_movie_dataset(**kwargs):
    R_true = np.load("data/rating_ml_100k_completed.npy")

    # normalize maximum value of R
    R_true = R_true / 5

    # rank of R
    rank = 10

    N, M = R_true.shape

    if kwargs.get("plot_data", False):
        plt.hist(R_true.flatten(), bins=np.int32(np.sqrt(N * M)))
        _, s, _ = np.linalg.svd(R_true)
        plt.yscale("log")
        plt.plot(s)
        plt.show()

    return R_true, rank


def get_dataset(dataset=None, **kwargs):
    if dataset == "synthetic":
        return get_synthetic_dataset(**kwargs)
    if dataset == "restaurant":
        return get_restaurant_dataset(**kwargs)
    if dataset == "movie":
        return get_movie_dataset(**kwargs)


def get_capacity(N, M, T, is_dynamic, dem_to_cap_ratio=1, p_activity=0.2):
    if is_dynamic:
        D = np.random.choice(2, size=(T, N), p=[1 - p_activity, p_activity])
        # set capacities randomly (changing with time)
        C_max = int(2 * dem_to_cap_ratio * np.ceil(np.sum(D[0]) / M))
        C = np.zeros((T, M), dtype=np.int64)
        C[0] = np.random.choice(C_max, size=(M)) + 1
        for t in range(1, T):
            C[t] = np.clip(C[t - 1] + np.random.choice(3, size=(M), p=[0.1, 0.8, 0.1]) - 1, 0, C_max)
    else:
        # set demands to 1 for all users
        p_activity = 1
        D = np.ones((T, N), dtype=np.int64)
        # set capacities randomly (fixed in time)
        C_max = int(2 * dem_to_cap_ratio * np.ceil(np.sum(D[0]) / M))
        C = np.zeros((T, M), dtype=np.int64)
        C[0] = np.random.choice(C_max, size=(M)) + 1
        for t in range(1, T):
            C[t] = C[t - 1]
    return C, D