from ortools.sat.python import cp_model
from ortools.linear_solver import pywraplp
from ortools.sat.python.cp_model import LinearExpr, CpSolver
from assign_problems import assign_matrix, proj_assign_matrix
from utility import create_tag_inputs
import numpy as np

def extreme_optional_vals(model, target, vars, var_exist):
    for var, ex in zip(vars, var_exist):
        model.Add(target >= var).OnlyEnforceIf(ex)

    equal_vars = []
    for var, ex in zip(vars, var_exist):
        equal_var = model.NewBoolVar("")
        model.Add(equal_var == 0).OnlyEnforceIf(ex.Not())
        model.Add(target == var).OnlyEnforceIf(equal_var)
        model.Add(target != var).OnlyEnforceIf(equal_var.Not())
        equal_vars.append(equal_var)

    model.Add(LinearExpr.Sum(equal_vars) >= 1)


model = cp_model.CpModel()

vars = [2,3,4,5,5,8,9]
exists = [model.NewBoolVar("") for _ in range(7)]
model.Add(LinearExpr.Sum(exists) == 4)
target = model.NewIntVar(0, 10, "")
extreme_optional_vals(model, target, vars, exists)
model.Minimize(target)

solver = cp_model.CpSolver()
status = solver.Solve(model)

print(solver.StatusName(status))

if status == 4:
    print(solver.Value(target))
    for i in exists:
        val = solver.Value(i)
        print(f"{i}th var {val}")

