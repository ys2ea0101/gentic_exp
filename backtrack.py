import numpy as np
from numba import njit
from assign_problems import assign_matrix
import time


def bound_estimation(m):
    p_min = np.sum(np.min(m, axis=0))
    w_min = np.sum(np.min(m, axis=1))
    return min(p_min, w_min)


def backtrack(p_path, w_path, n, current_val, p_avail, w_avail, cost_matrix, best):
    """

    :param p_path: project path, list
    :param w_path: worker path, list
    :param n: total p or w, int
    :param current_val: current sum, int
    :param p_avail: leftover project, list
    :param w_avail: leftover worker, list
    :param cost_matrix:
    :param best:
    :return:
    """
    if len(p_path) == n:
        best[0] = min(best[0], current_val)
        calc = np.sum(cost_matrix[w_path, p_path])
        print(f"{w_path = }, {p_path = }, {current_val = }, {calc = }")
        return
    m_sliced = cost_matrix[w_avail, :][:, p_avail]
    bound = bound_estimation(m_sliced)
    if current_val + bound >= best[0]:
        return
    guess = np.unravel_index(np.argmin(m_sliced, axis=None), m_sliced.shape)
    p_select = p_avail[guess[1]]
    w_order = np.argsort(m_sliced[:, guess[1]])
    w_vals = np.sort(m_sliced[:, guess[1]])
    n_p = [p for p in p_avail if p != p_select]
    for w, wv in zip(w_order, w_vals):
        backtrack(p_path + [p_select], w_path + [w_avail[w]], n, current_val + wv, n_p, [ww for ww in w_avail if ww != w_avail[w]], cost_matrix, best)


def cp_solver(cost_matrix):
    best = [1000]
    l = cost_matrix.shape[0]
    backtrack([], [], l, 0, list(range(l)), list(range(l)), cost_matrix, best)
    return best[0]


if __name__ == "__main__":
    mtx = assign_matrix(10, 10, 50, seed=345)
    print(mtx)
    t0 = time.perf_counter()
    slt = cp_solver(mtx.transpose())
    t1 = time.perf_counter()
    print(f"Result: {slt}, time used: {t1 - t0}")

    from cp_mip_solver import run_linear_assignment_solver
    run_linear_assignment_solver(mtx, "cp")

