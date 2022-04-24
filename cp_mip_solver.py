from ortools.sat.python import cp_model
from ortools.linear_solver import pywraplp
from ortools.sat.python.cp_model import LinearExpr, CpSolver
from assign_problems import assign_matrix, proj_assign_matrix
from utility import create_tag_inputs
import numpy as np
import time

def run_linear_assignment_solver(cost, solver):
    if solver == "mip":
        model = pywraplp.Solver.CreateSolver('SCIP')
    elif solver == "cp":
        model = cp_model.CpModel()
    else:
        raise ValueError("model not recognized")

    num_workers = cost.shape[0]
    num_tasks = cost.shape[1]
    # Variables
    x = []
    for i in range(num_workers):
        t = []
        for j in range(num_tasks):
            t.append(model.NewBoolVar(f'x[{i},{j}]'))
        x.append(t)

    # Constraints
    # Each worker is assigned to at most one task.
    for i in range(num_workers):
        model.Add(LinearExpr.Sum([x[i][j] for j in range(num_tasks)]) <= 1)

    # Each task is assigned to exactly one worker.
    for j in range(num_tasks):
        model.Add(LinearExpr.Sum([x[i][j] for i in range(num_workers)]) == 1)

    # Objective
    objective_terms = []
    for i in range(num_workers):
        for j in range(num_tasks):
            objective_terms.append(cost[i][j] * x[i][j])
    model.Minimize(sum(objective_terms))

    # Solve
    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    print("Solve status: %s" % solver.StatusName(status))
    print("Optimal objective value: %i" % solver.ObjectiveValue())
    print("conflicts : %i" % solver.NumConflicts())
    print("branches  : %i" % solver.NumBranches())

    # Print solution.
    print_solution = False
    if print_solution:
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            print(f'Total cost = {solver.ObjectiveValue()}')
            print()
            for i in range(num_workers):
                for j in range(num_tasks):
                    if solver.BooleanValue(x[i][j]):
                        print(
                            f'Worker {i} assigned to task {j} Cost = {cost[i][j]}')
        else:
            print('No solution found.')


def run_abs_assignment_solver(worker_tag, proj_tag):

    def extreme_optional_vals(model, target, vars, var_exist):
        for var, ex in zip(vars, var_exist):
            model.Add(target >= var).OnlyEnforceIf(ex)

        equal_vars = []
        for var, ex in zip(vars, var_exist):
            equal_var = model.NewBoolVar("")
            model.Add(equal_var == 1).OnlyEnforceIf(ex)
            model.Add(target == var).OnlyEnforceIf(equal_var)
            model.Add(target != var).OnlyEnforceIf(equal_var.Not())
            equal_vars.append(equal_var)

        model.Add(LinearExpr.Sum(equal_vars) >= 1)

    model = cp_model.CpModel()

    num_workers = len(worker_tag)
    num_tasks = len(proj_tag)
    if isinstance(worker_tag, np.ndarray) and worker_tag.ndim > 1:
        D = worker_tag.shape[1]
    else:
        D = 1
    # Variables
    x = []
    for i in range(num_workers):
        t = []
        for j in range(num_tasks):
            t.append(model.NewBoolVar(f'x[{i},{j}]'))
        x.append(t)

    # Constraints
    # Each worker is assigned to at most one task.
    for i in range(num_workers):
        model.Add(LinearExpr.Sum([x[i][j] for j in range(num_tasks)]) <= 1)

    for j in range(num_tasks):
        model.Add(LinearExpr.Sum([x[ii][j] for ii in range(num_workers)]) <= 3)

    # Objective, abs diff
    objective_terms = []
    for j in range(num_tasks):
        worker_sum = model.NewIntVar(-30, 30, "")
        abs_diff = model.NewIntVar(0, 30, "")
        for dd in range(D):
            model.Add(worker_sum == LinearExpr.Sum([worker_tag[i, dd] * x[i][j] for i in range(num_workers)]) - proj_tag[j, dd])
            model.AddAbsEquality(abs_diff, worker_sum)
            objective_terms.append(abs_diff)
    model.Minimize(sum(objective_terms))

    # Objective, max
    # objective_terms = []
    # for j in range(num_tasks):
    #     max_worker_val = model.NewIntVar(0, 20, "")
    #
    #     extreme_optional_vals(model, max_worker_val, list(worker_tag), [x[kk][j] for kk in range(num_workers)])
    #
    #     for ii in range(num_workers):
    #         var = model.NewIntVar(0, 100, "")
    #         multi_var = model.NewIntVar(0, 100, "")
    #         model.Add(var == max_worker_val + worker_tag[ii])
    #         model.AddMultiplicationEquality(multi_var, [var, x[ii][j]])
    #         objective_terms.append(multi_var)
    # model.Maximize(sum(objective_terms))

    # Solve
    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    print(f"status: {status}")
    print("conflicts : %i" % solver.NumConflicts())
    print("branches  : %i" % solver.NumBranches())

    # Print solution.
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print(f'Total cost = {solver.ObjectiveValue()}')
        print()
        for i in range(num_workers):
            for j in range(num_tasks):
                if solver.BooleanValue(x[i][j]):
                    print(
                        f'Worker {i} assigned to task {j}')
    else:
        print('No solution found.')


if __name__ == "__main__":
    n_ppl = 100
    n_proj = 100
    cost = assign_matrix(n_ppl, n_proj, 20)
    to = time.perf_counter()
    run_linear_assignment_solver(cost, "cp")
    t1 = time.perf_counter()
    # people_tag = np.random.randint(0, 5, size=(n_ppl, ))
    # ppl_select = np.arange(n_ppl)
    # np.random.shuffle(ppl_select)
    # solution = np.zeros((n_ppl,), dtype=int)
    # proj_tag = []
    # for i in range(n_proj):
    #     t = 4
    #     for k in range(i * 3, i * 3 + 3):
    #         t = t + people_tag[ppl_select[k]]
    #         solution[ppl_select[k]] = i + 1
    #     proj_tag.append(t)
    # proj_tag = np.array(proj_tag)
    # print(people_tag, proj_tag)
    # run_abs_assignment_solver(people_tag, proj_tag)


    # ret = create_tag_inputs(n_ppl, n_proj, 5, manipulate_proj=True)
    # run_abs_assignment_solver(ret["people_tag"], ret["project_tag"])


