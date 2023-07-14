from ortools.algorithms import pywrapknapsack_solver
import time

# Create the solver.
solver = pywrapknapsack_solver.KnapsackSolver(
    pywrapknapsack_solver.KnapsackSolver.
    KNAPSACK_MULTIDIMENSION_BRANCH_AND_BOUND_SOLVER, 'KnapsackExample')

def knapsack(data,timelimit):
    solver.Init(data["values"],data["weights"],data["capacities"])
    solver.set_time_limit(timelimit)
    time_start = time.time()
    total_value = solver.Solve()
    time_end = time.time()
    runtime = time_end - time_start
    isOptimal = (runtime <= timelimit)
    
    packed_items = []
    packed_weights = []
    total_weight = 0
    for i in range(len(data["values"])):
        if solver.BestSolutionContains(i):
            packed_items.append(i)
            packed_weights.append(data["weights"][0][i])
            total_weight += data["weights"][0][i]
            total_value +=data["values"][i]
    return {"packed_items":packed_items,
            "packed_weights":packed_weights,
            "total_weight":total_weight,
            "total_value":total_value,
            "isOptimal":isOptimal,
            "runtime":runtime
            }