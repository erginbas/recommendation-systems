import numpy as np
from PyomoSolver import PyomoSolver
import time
import logging

class Algorithms:
    def __init__(self, R_true, rank, eta, is_dynamic, C, D, T, exp_save_path, verbose = False):
        self.R_true = R_true
        self.rank = rank
        self.N, self.M = R_true.shape
        self.eta = eta
        self.T = T
        self.is_dynamic = is_dynamic
        self.C = C
        self.D = D
        self.exp_save_path = exp_save_path

        logging.basicConfig(filename=f"{exp_save_path}/log.out",
                            filemode='a',
                            format='[%(asctime)s] %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.INFO)

        # shift parameter for prices (only to be used if users accept/reject)
        self.nu = 0.5 * (((self.N + self.M) * self.rank * self.eta ** 2)/(self.N * self.M * self.T))**(1/4)

        logging.info(f"Price shift = {self.nu}")

        self.initial_mask = None
        self.opt_rewards = None

        # use a MIP solver to calculate optimal allocations efficiently
        self.solver = PyomoSolver(self.N, self.M)
        self.find_optimum()
        self.generate_initial_mask()



    def find_optimum(self):
        # calculate optimal allocations for each time t
        self.x_star = np.zeros((self.T, self.N, self.M))
        self.opt_rewards = np.zeros(self.T)

        # solve for optimum allocations
        if self.is_dynamic:
            for t in range(self.T):
                self.x_star[t] = self.solver.solve_system(self.R_true, self.C[t], self.D[t])
                self.opt_rewards[t] = np.sum(self.x_star[t] * self.R_true)
                if t % 10 == 0:
                    logging.info(f'solved x_star at {t}, optimum value = {np.sum(self.x_star[t] * self.R_true)}')
        else:
            for t in range(self.T):
                if t == 0:
                    self.x_star[0] = self.solver.solve_system(self.R_true, self.C[0], self.D[0])
                    self.opt_rewards[t] = np.sum(self.x_star[t] * self.R_true)
                    logging.info(f'solved x_star, optimum value = {np.sum(self.x_star[t] * self.R_true)}')
                else:
                    self.x_star[t] = self.x_star[t - 1]
                    self.opt_rewards[t] = np.sum(self.x_star[t] * self.R_true)

        logging.info(f"Optimum prices = {self.solver.get_prices()}")

        np.save(f"{self.exp_save_path}/opt_rewards", self.opt_rewards)

    def calculate_instability(self, t, X, p):
        best_surplus_list = np.zeros(self.N)
        total_current_surplus = np.sum(X * (self.R_true - p))

        for u in range(self.N):
            demand_u = self.D[t, u]
            ordered_items = np.maximum(np.sort(self.R_true[u, :] - p), 0)
            best_surplus_list[u] = np.sum(ordered_items[-demand_u:])

        return np.sum(best_surplus_list) - total_current_surplus

    def calculate_regret(self, t, X, p_walrasian, nu):
        sw_regret = self.opt_rewards[t] - np.sum(X * self.R_true)
        instability = self.calculate_instability(t, X, p_walrasian)

        p_shifted = p_walrasian - nu
        accepted_offers = X * (self.R_true > p_shifted)
        sw_regret_ar = self.opt_rewards[t] - np.sum(accepted_offers * self.R_true)
        instability_ar = self.calculate_instability(t, accepted_offers, p_shifted)

        return [sw_regret, instability, sw_regret_ar, instability_ar]


    def generate_initial_mask(self):
        # generate initial mask
        p = 0.06
        is_mask_feasible = False
        while not is_mask_feasible:
            initial_mask = np.random.binomial(1, p, size=(self.N, self.M))
            p += 0.01
            if np.min(np.sum(initial_mask, axis=0)) > 0 and np.min(np.sum(initial_mask, axis=1)) > 0:
                is_mask_feasible = True
                self.initial_mask = initial_mask

    def solve_algo(self, algorithm):
        if algorithm == "CUCB":
            return self.solve_CUCB()
        elif algorithm == "OFU":
            return self.solve_OFU()
        elif algorithm == "ACF":
            return self.solve_ACF()
        elif algorithm == "ICF":
            return self.solve_ICF()
        elif algorithm == "ICF2":
            return self.solve_ICF2()
        else:
            raise NotImplementedError

    def solve_CUCB(self):
        """### CUCB (Combinatorial UCB for CMAB without structure)"""

        alpha = 2
        data_sum = np.zeros((self.N, self.M))
        data_counts = np.zeros((self.N, self.M))

        def update_ucb(observations):
            for obs in observations:
                data_sum[obs[0], obs[1]] += obs[2]
                data_counts[obs[0], obs[1]] += 1

            data_sum[data_counts == 0] = 1
            data_counts[data_counts == 0] = 1

            R_UCB = data_sum / data_counts
            std = alpha / np.sqrt(data_counts)

            return R_UCB + std

        regrets = np.zeros((self.T, 4))

        # get initial observations
        observations = self.get_observations_gauss(self.initial_mask)

        x_prev = None

        for t in range(self.T):

            R_UCB = update_ucb(observations)

            x_UCB = self.solver.solve_system(R_UCB, self.C[t], self.D[t], x_prev=x_prev)
            x_prev = x_UCB.copy()

            p_wal = self.solver.get_prices()
            regret_collection = self.calculate_regret(t, x_UCB, p_wal, self.nu)
            regrets[t] = regret_collection

            if t % 10 == 0:
                logging.info(f'Iter {t}')
                logging.info(f'Social Welfare Regret = {regrets[t, 0]}')
                logging.info(f'Instability = {regrets[t, 1]}')
                logging.info(f'Social Welfare Regret (A/R) = {regrets[t, 2]}')
                logging.info(f'Instability (A/R) = {regrets[t, 3]}')

            observations = self.get_observations_gauss(x_UCB)

        self.save_results("CUCB", regrets)
        return regrets

    def solve_OFU(self):
        """our proposed algorithm, OFU with low-rank collaborative filtering"""

        # get initial observations
        observations = self.get_observations_gauss(self.initial_mask)

        regrets = np.zeros((self.T, 4))
        x_prev = None

        for t in range(self.T):
            U = np.random.randn(self.N, self.rank)
            V = np.random.randn(self.M, self.rank)

            U, V = self.compute_center(observations, U, V)
            counts = self.update_counts(observations)

            x_OFU = self.solve_for_ofu_allocation(U, V, counts, t, x_prev=x_prev)
            x_prev = x_OFU.copy()

            p_wal = self.solver.get_prices()
            regret_collection = self.calculate_regret(t, x_OFU, p_wal, self.nu)
            regrets[t] = regret_collection

            if t % 10 == 0:
                logging.info(f'Iter {t}')
                logging.info(f'Social Welfare Regret = {regrets[t, 0]}')
                logging.info(f'Instability = {regrets[t, 1]}')
                logging.info(f'Social Welfare Regret (A/R) = {regrets[t, 2]}')
                logging.info(f'Instability (A/R) = {regrets[t, 3]}')

            observations += self.get_observations_gauss(x_OFU)

        self.save_results("OFU", regrets)
        return regrets

    def solve_ACF(self):
        """ ACF (Low-rank collaborative filtering without exploration, best allocation w.r.t. LS estimate)"""
        # get initial observations
        observations = self.get_observations_gauss(self.initial_mask)

        regrets = np.zeros((self.T, 4))

        x_prev = None
        U = np.random.randn(self.N, self.rank)
        V = np.random.randn(self.M, self.rank)

        for t in range(self.T):

            U, V = self.compute_center(observations, U, V)
            counts = self.update_counts(observations)
            R_est = U @ V.T

            x_t = self.solver.solve_system(R_est, self.C[t], self.D[t], x_prev=x_prev)
            x_prev = x_t.copy()

            p_wal = self.solver.get_prices()
            regret_collection = self.calculate_regret(t, x_t, p_wal, self.nu)
            regrets[t] = regret_collection

            if t % 10 == 0:
                logging.info(f'Iter {t}')
                logging.info(f'Social Welfare Regret = {regrets[t, 0]}')
                logging.info(f'Instability = {regrets[t, 1]}')
                logging.info(f'Social Welfare Regret (A/R) = {regrets[t, 2]}')
                logging.info(f'Instability (A/R) = {regrets[t, 3]}')

            observations += self.get_observations_gauss(x_t)

        self.save_results("ACF", regrets)
        return regrets


    def solve_ICF(self):

        # get initial observations
        observations = self.get_observations_gauss(self.initial_mask)

        regrets = np.zeros((self.T, 4))

        x_prev = None

        for t in range(self.T):

            U = np.random.randn(self.N, self.rank)
            V = np.random.randn(self.M, self.rank)

            U, V = self.compute_center(observations, U, V)
            counts = self.update_counts(observations)

            # obtain solution without capacity constraints
            x_OFU = self.solve_for_ofu_allocation(U, V, counts, t, x_prev=x_prev, solve_with_capacity=False)

            # only the users with high reward obtain the items if there are too many requests for the same item
            for i in range(self.M):
                while np.sum(x_OFU[:, i]) > self.C[t, i]:
                    allocated_users = x_OFU[:, i] > 0.9
                    temp = self.R_true[:, i].copy()
                    temp[~allocated_users] = np.inf
                    v = np.argmin(temp)
                    x_OFU[v, i] = 0

            p_wal = self.solver.get_prices()
            regret_collection = self.calculate_regret(t, x_OFU, p_wal, self.nu)
            regrets[t] = regret_collection

            if t % 10 == 0:
                logging.info(f'Iter {t}')
                logging.info(f'Social Welfare Regret = {regrets[t, 0]}')
                logging.info(f'Instability = {regrets[t, 1]}')
                logging.info(f'Social Welfare Regret (A/R) = {regrets[t, 2]}')
                logging.info(f'Instability (A/R) = {regrets[t, 3]}')

            observations += self.get_observations_gauss(x_OFU)

        self.save_results("ICF", regrets)
        return regrets


    def solve_ICF2(self):

        # get initial observations
        observations = self.get_observations_gauss(self.initial_mask)

        regrets = np.zeros((self.T, 4))

        x_prev = None

        for t in range(self.T):

            U = np.random.randn(self.N, self.rank)
            V = np.random.randn(self.M, self.rank)

            U, V = self.compute_center(observations, U, V)
            counts = self.update_counts(observations)

            # obtain solution without capacity constraints
            x_OFU_intended = self.solve_for_ofu_allocation(U, V, counts, t, x_prev=x_prev, solve_with_capacity=False)
            x_OFU = x_OFU_intended.copy()


            # only some of the users obtain the items if there are too many requests for the same item
            for i in range(self.M):
                while np.sum(x_OFU[:, i]) > self.C[t, i]:
                    allocated_users = x_OFU[:, i] > 0.9
                    v = np.random.choice(np.where(allocated_users)[0])
                    x_OFU[v, i] = 0

            x_OFU_uns = x_OFU_intended - x_OFU

            p_wal = self.solver.get_prices()
            regret_collection = self.calculate_regret(t, x_OFU, p_wal, self.nu)
            regrets[t] = regret_collection

            if t % 10 == 0:
                logging.info(f'Iter {t}')
                logging.info(f'Social Welfare Regret = {regrets[t, 0]}')
                logging.info(f'Instability = {regrets[t, 1]}')
                logging.info(f'Social Welfare Regret (A/R) = {regrets[t, 2]}')
                logging.info(f'Instability (A/R) = {regrets[t, 3]}')

            observations += self.get_observations_gauss(x_OFU_intended)
            observations + self.get_zero_observations(x_OFU_uns)

        self.save_results("ICF2", regrets)
        return regrets

    def compute_center(self, observations, U, V):
        k = self.rank
        eps = 1e-2

        iter = 0

        obs_items = [[] for _ in range(self.N)]
        obs_users = [[] for _ in range(self.M)]

        for z, obs in enumerate(observations):
            obs_items[obs[0]].append(z)
            obs_users[obs[1]].append(z)

        while True:
            U_prev = U.copy()
            for i in range(self.N):
                Sigma_sum = 0
                mean_sum = np.zeros(k)
                for z in obs_items[i]:
                    obs = observations[z]
                    Sigma_sum += np.outer(V[obs[1]], V[obs[1]])
                    mean_sum += obs[2] * V[obs[1]]
                U[i] = np.linalg.inv(Sigma_sum + eps * np.eye(k)) @ mean_sum

            for j in range(self.M):
                Sigma_sum = 0
                mean_sum = np.zeros(k)
                for z in obs_users[j]:
                    obs = observations[z]
                    Sigma_sum += np.outer(U[obs[0]], U[obs[0]])
                    mean_sum += obs[2] * U[obs[0]]
                V[j] = np.linalg.inv(Sigma_sum + eps * np.eye(k)) @ mean_sum

            iter += 1
            if np.linalg.norm(U - U_prev) < np.maximum(self.N, self.M) * k * 1e-2 or iter > 10:
                break

        return U, V

    def update_counts(self, observations):
        counts = np.zeros((self.N, self.M)) + 0.01
        for obs in observations:
            counts[obs[0], obs[1]] += 1
        return counts

    def solve_for_ofu_allocation(self, U, V, counts, t, x_prev=None, solve_with_capacity=True):
        beta = 8 * (self.eta ** 2) * (self.N + self.M + 1) * self.rank * np.log(9 * 16 * (self.N ** 2) * self.M * self.T)
        beta += 8 * (self.eta ** 2) * np.log(self.T)
        beta += 16 * t / self.T + 2 * np.sqrt(8 * (self.eta ** 2) * np.log(4 * (self.N ** 2) * (self.T ** 3)))

        beta = beta * 1e-2

        Theta_LS = U @ V.T

        x = np.ones((self.N, self.M))
        alpha_UV_init = 1e-2

        def check_conf(A, B):
            diff = A @ B.T - Theta_LS
            return np.sum(counts * diff ** 2) <= beta

        def opt_UV(U, V, x):
            grad_U = x @ V
            alpha_UV = alpha_UV_init
            while True:
                U_temp = U + alpha_UV * grad_U
                inside = check_conf(U_temp, V)
                if inside:
                    U = U_temp
                    break
                else:
                    alpha_UV = alpha_UV / 2

            grad_V = x.T @ U
            alpha_UV = alpha_UV_init
            while True:
                V_temp = V + alpha_UV * grad_V
                inside = check_conf(U, V_temp)
                if inside:
                    V = V_temp
                    break
                else:
                    alpha_UV = alpha_UV / 2
            return U, V

        if solve_with_capacity:
            U_prev = np.zeros_like(U)
            for i in range(3):
                for v in range(10):
                    if np.linalg.norm(U - U_prev) < 1e-6:
                        continue
                    U_prev = U.copy()
                    U, V = opt_UV(U, V, x)
                    if np.linalg.norm(U - U_prev) < 1e-6:
                        break

                R = U @ V.T

                x = self.solver.solve_system(R, self.C[t], self.D[t], x_prev=x_prev)
        else:
            for v in range(10):
                U_prev = U.copy()
                U, V = opt_UV(U, V, x)
                if np.linalg.norm(U - U_prev) < 1e-6:
                    break
            R = U @ V.T
            C_inf = np.ones_like(self.C[t]) * self.N
            x = self.solver.solve_system(R, C_inf, self.D[t], x_prev=x_prev)

        return x

    def get_observations_gauss(self, X):
        observations = []
        for i in range(self.N):
            for j in range(self.M):
                if X[i, j] == 1:
                    observations.append((i, j, self.R_true[i, j] + self.eta * np.random.randn()))
        return observations

    def get_zero_observations(self, X):
        observations = []
        for i in range(self.N):
            for j in range(self.M):
                if X[i, j] == 1:
                    observations.append((i, j, 0))
        return observations

    def save_results(self, algo, regrets):
        np.save(f"{self.exp_save_path}/{algo}_regrets", regrets)
