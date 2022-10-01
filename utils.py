import numpy as np
import time

from cvxopt import matrix, solvers, log, exp
from cvxopt.glpk import ilp

from ortools.linear_solver import pywraplp
from ortools.sat.python import cp_model

method = "cvxopt"

def solve_system(*args, **kwargs):
    start = time.time()
    if method == "distributed":
        sol = solve_system_distributed(*args, **kwargs)
    elif method == "ortools":
        sol = solve_system_centralized_ortools(*args, **kwargs)
    elif method == "cxvopt":
        sol = solve_system_centralized_cvxopt(*args, **kwargs)
    else:
        raise NotImplementedError
    end = time.time()
    print("elapsed time = ", end - start)
    return sol


def solve_system_centralized_ortools(R, C, D, verbose=True):
    N, M = R.shape

    model = cp_model.CpModel()

    x = []
    for i in range(N):
        t = []
        for j in range(M):
            t.append(model.NewBoolVar(f'x[{i},{j}]'))
        x.append(t)

    for i in range(N):
        model.Add(sum([x[i][j] for j in range(M)]) <= D[i])

    for j in range(M):
        model.Add(sum([x[i][j] for i in range(N)]) <= C[j])

    objective_terms = []
    for i in range(N):
        for j in range(M):
            objective_terms.append(R[i, j] * x[i][j])
    model.Maximize(sum(objective_terms))

    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    x_sol = np.zeros((N, M))

    for i in range(N):
        for j in range(M):
            if solver.BooleanValue(x[i][j]):
                x_sol[i, j] = 1

    if verbose:
        print('Problem solved in %f seconds' % solver.WallTime())
        print('Problem solved in %d branch-and-bound branches' % solver.NumBranches())

    return x_sol


# def solve_system_2(R, C, D, verbose=False):
#     N, M = R.shape
#
#     solver = pywraplp.Solver.CreateSolver('SCIP')
#
#     x = {}
#
#     for i in range(N):
#         for j in range(M):
#             x[i, j] = solver.IntVar(0, 1, '')
#
#     for i in range(N):
#         solver.Add(solver.Sum([x[i, j] for j in range(M)]) <= D[i])
#
#     for j in range(M):
#         solver.Add(solver.Sum([x[i, j] for i in range(N)]) <= C[j])
#
#     objective_terms = []
#     for i in range(N):
#         for j in range(M):
#             objective_terms.append(R[i, j] * x[i, j])
#     solver.Maximize(solver.Sum(objective_terms))
#
#     status = solver.Solve()
#
#     x_sol = np.zeros((N, M))
#
#     for i in range(N):
#         for j in range(M):
#             # Test if x[i,j] is 1 (with tolerance for floating point arithmetic).
#             if x[i, j].solution_value() > 0.5:
#                 x_sol[i, j] = 1
#
#     if verbose:
#         if status == pywraplp.Solver.OPTIMAL:
#             print('Objective value =', solver.Objective().Value())
#             print('Problem solved in %f milliseconds' % solver.wall_time())
#             print('Problem solved in %d iterations' % solver.iterations())
#             print('Problem solved in %d branch-and-bound nodes' % solver.nodes())
#         else:
#             print('The problem does not have an optimal solution.')
#
#     return x_sol


def solve_system_centralized_cvxopt(R, C, D):
    N, M = R.shape
    A1 = np.zeros((N, N * M))
    for i in range(N):
        A1[i, i * M:(i + 1) * M] = np.ones((M))

    A2 = np.zeros((M, N * M))
    for i in range(M):
        A2[i, i:N * M:M] = np.ones((N))

    A = np.concatenate((A1, A2))

    c = -1 * R.reshape((N * M, 1)) + 1e-5 * np.random.randn((N * M), 1)

    b = np.concatenate((D, C))

    solvers.options['show_progress'] = False
    (status, x) = ilp(matrix(c,  tc='d'), matrix(A, tc='d'), matrix(b, tc='d'), B=set(range(N * M)))

    return np.int32(np.reshape(x, (N, M)))


def solve_system_distributed(R, C, D, alpha = 1e-3):
    N, M = R.shape

    mu = np.ones(M) * np.max(R) / 2
    x = np.zeros((N, M))

    k = 0

    while True:

        mu_prev = mu
        x_prev = x

        U_eff = R - mu
        x = np.zeros((N, M))

        for u in range(N):
            demand = D[u]
            if demand > 0:
                selected_items = np.argpartition(U_eff[u, :], -demand)[-demand:]
                x[u, selected_items] = 1

        x = x * (U_eff > 0)

        mu = np.maximum(mu - alpha/((k+1) ** 0.5) * (C - np.sum(x, axis=0)), 0)

        if k % 500 == 0:
            print(k, np.linalg.norm(mu - mu_prev), np.sum(np.abs(x - x_prev)), alpha/((k+1) ** 0.2))

        k = k+1

        if k > 3000 or (np.linalg.norm(mu - mu_prev)/N < 1e-7 and np.sum(np.abs(x - x_prev)) < 1):
            return np.int32(x > 0.9)


is_initialized = False

