
import numpy as np
import gurobipy as gp
from gurobipy import GRB, quicksum
from tqdm.notebook import trange
import argparse


np.set_printoptions(suppress=True)

from constants import (
    MAX_TAXI_ZONE_ID,
    location_ids,
    excluded_location_ids,
    location_id_to_index,
    num_locations,
    taxi_type,
)

def compute_time_averaged(lambda_, mu, t_start, K, T_max):
    time_indices = [(t_start + k) % T_max for k in range(K)]
    lambda_avg = np.mean(lambda_[time_indices, :], axis=0)         # shape: (r,)
    mu_avg = np.mean(mu[time_indices, :, :], axis=0)               # shape: (r, r)
    return lambda_avg, mu_avg


def solve_routing(lambda_, mu, P, Delta, T, t_start=0, K=6):
    lambda_avg, mu_avg = compute_time_averaged(lambda_, mu, t_start, K, T)
    P_avg = np.mean(P[[(t_start + k) % T for k in range(K)], :, :], axis=0)  # optional if P is time-varying

    r = lambda_.shape[1]
    model = gp.Model("lookahead_empty_car_routing")
    model.setParam('OutputFlag', 0)
    
    # For numeric stability
    
    # model.setParam('OptimalityTol', 1e-9)
    # model.setParam('FeasibilityTol', 1e-9)
    # model.setParam('NumericFocus', 3)
    
    # scaling e, f, and a by this factor for numeric stability
    SCALE = 1000
    
    # Variables
    a = model.addVars(r, lb=0, ub=1, name="a")
    e = model.addVars(r, r, lb=0, ub=SCALE, name="e")
    f = model.addVars(r, r, lb=0, ub=SCALE, name="f")
    # hello = model.addVar(lb=0, ub=1, name="hello")

    # Objective
    model.setObjective(
        quicksum(a[i] * lambda_avg[i] * P_avg[i, j] for i in range(r) for j in range(r)),# + 1 * hello,
        GRB.MAXIMIZE
    )

    # # fairness
    # model.addConstrs(
    #     a[i] >= hello for i in range(r)
    # )
        

    # Eq. 10a: Ride flow balance
    ride_flow_expr = {
        (i, j): SCALE * a[i] * lambda_avg[i] * P_avg[i, j] - mu_avg[i, j] * f[i, j]
        for i in range(r) for j in range(r)
    }
    model.addConstrs((ride_flow_expr[i, j] == 0 for i in range(r) for j in range(r)), name="ride_flow_balance")

    # Eq. 10b: Empty car flow balance
    model.addConstrs((
        mu_avg[i, j] * e[i, j] <= quicksum(mu_avg[l, i] * f[l, i] for l in range(r))
        for i in range(r) for j in range(r) if i != j
    ), name="empty_car_flow")

    # Eq. 10c: Supply conservation (lower bound)
    model.addConstrs((
        quicksum(mu_avg[j, i] * e[j, i] for j in range(r) if j != i) <= SCALE * lambda_avg[i] * a[i]
        for i in range(r)
    ), name="supply_lower")
    
    # Eq. 10c: Supply conservation (upper bound)
    model.addConstrs((
        SCALE * lambda_avg[i] * a[i] <=
        quicksum(mu_avg[j, i] * e[j, i] for j in range(r) if j != i) +
        quicksum(mu_avg[j, i] * f[j, i] for j in range(r))
        for i in range(r)
    ), name="supply_upper")
    
    # Eq. 10d: Car flow balance
    model.addConstrs((
        SCALE * lambda_avg[i] * a[i] +
        quicksum(mu_avg[i, j] * e[i, j] for j in range(r) if j != i)
        ==
        quicksum(mu_avg[j, i] * e[j, i] for j in range(r) if j != i) +
        quicksum(mu_avg[j, i] * f[j, i] for j in range(r))
        for i in range(r)
    ), name="car_flow_balance")
    

    # Unit mass constraint
    model.addConstr(
        quicksum(e[i,j] + f[i,j] for i in range(r) for j in range(r)) == SCALE,
        name="unit_mass"
    )
    
    model.optimize()
    
    return model, a, e, f, lambda_avg, mu_avg

def compute_q_matrix(a, e, f, mu_avg, lambda_avg):
    """
    a, e, f: 2D array of shape (r, r)
    mu_avg: 2D array of shape (r, r) averaged over time
    lambda_avg: 1D array of shape (r,) averaged over time
    """
    r = len(a)
    q = np.zeros((r, r))

    # Precompute denominator for qij and qii
    for i in range(r):
        denom = sum(mu_avg[k, i] * f[k, i] for k in range(r))
        # print(f"Denominator is zero for i={i}. Setting q[{i}, :] to 0.")
        for j in range(r):
            if i != j:
                if denom > 0:
                    q[i, j] = mu_avg[i, j] * e[i, j] / denom
                else:
                    
                    q[i, j] = 0.0
            else:
                # qii computation
                numerator = lambda_avg[i] * a[i] - sum(
                    mu_avg[k, i] * e[k, i] for k in range(r) if k != i
                )
                q[i, i] = numerator / denom if denom > 0 else 0.0

    return q

def solve_Q(Delta, P, lambda_, mu, K, T):
    Qs = []
    for t_start in trange(T):
        print(f"t_start: {t_start}")
        model, a, e, f, lambda_avg, mu_avg = solve_routing(lambda_, mu, P, Delta, T, t_start=t_start, K=K)
        
        # Extract the values of the variables
        a_values = np.array([a[i].X for i in range(R)])
        e_values = np.array([[e[i,j].X for j in range(R)] for i in range(R)])
        f_values = np.array([[f[i,j].X for j in range(R)] for i in range(R)])

        normalizer = (e_values + f_values).sum()

        e_values /= normalizer
        f_values /= normalizer
            
        # Compute the Q matrix
        Q = compute_q_matrix(a_values, e_values, f_values, mu_avg, lambda_avg)
        
        # clip values close to 0
        Q[np.isclose(Q, 0, atol=1e-9)] = 0
        # normalize rows of Q to sum to 1
        Q = Q / Q.sum(axis=1, keepdims=True)
        
        Qs.append(Q)
        
        # check rows are nonnegative / no entries with absolute value > 1
        if not np.all(Q >= 0):
            print(f"Q matrix has negative entries at t_start={t_start}")
        if not np.all(np.abs(Q) <= 1):
            print(f"Q matrix has entries with absolute value > 1 at t_start={t_start}")
            
    # save Qs as len(Q) x r x r np.ndarray
    Qs = np.array(Qs)
    np.savez(f'Qs_{K}_clipping_2.npz', Qs=Qs)
    return Qs


if __name__ == "__main__":
    # use argparse to read parameter L, an integer
    parser = argparse.ArgumentParser(description='Solve the Q matrix.')
    parser.add_argument('--L', type=int, help='Lookahead parameter')
    L = parser.parse_args().L
    
    # print the value of L
    print(f"L: {L}")
    # Usage: python solve_relocation_matrix.py --L 4
    
    # Create a Gurobi model
    Delta = 20 # in minutes

    # Prepare System Parameters
    with np.load('trip_counts.npz') as data:
        trip_counts = data['trip_counts']
        num_dates = data['num_dates']

    with np.load('mu_cp_clipped.npz') as data:
        mu = data['mu']
        
    # mask trip_counts by 1 where 0
    trip_counts[trip_counts == 0] = 1

    # compute arrival rate
    lambda_ = trip_counts.sum(axis=2) / (Delta / 60 * num_dates)
    lambda_2 = lambda_ / 8000

    # normalize trip_counts to get transition probabilities
    P = trip_counts / trip_counts.sum(axis=2, keepdims=True)

    T, R, _ = P.shape
    
    solve_Q(Delta, P, lambda_2, mu, L, T)