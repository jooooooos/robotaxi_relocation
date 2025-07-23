import numpy as np
from constants import IS_IDLE, IS_RELOCATING

def relocation_policy_blind_sampling(vehicle, current_time, simulator, **kwargs):
    """Relocate based on blind sampling from Q[t]."""
    t_index = int(current_time) % simulator.Q.shape[0]
    probs = simulator.Q[t_index, vehicle.location]
    destination = np.random.choice(len(probs), p=probs)
    return destination


def relocation_policy_jlcr_eta(vehicle, current_time, simulator, eta=0.5, **kwargs):
    """Relocate based on Join-the-Least-Congested-Region (JLCR-eta) heuristic."""
    t_index = int(current_time) % simulator.lambda_.shape[0]
    R = simulator.lambda_.shape[1]

    # Count idle and relocating cars toward each region
    idle_relocating_counts = np.zeros(R)
    for v in simulator.vehicles:
        if v.status == IS_IDLE:
            idle_relocating_counts[v.location] += 1
        elif v.status == IS_RELOCATING and v.target_location is not None:
            idle_relocating_counts[v.target_location] += 1

    congestion = idle_relocating_counts / (simulator.lambda_[t_index] + 1e-6)

    current_congestion = congestion[vehicle.location]
    congestion_wo_current = np.delete(congestion, vehicle.location)

    min_congestion = np.min(congestion_wo_current)
    best_region = np.argmin(congestion_wo_current)
    if best_region >= vehicle.location:
        best_region += 1

    if (1 - eta) * current_congestion > min_congestion:
        return best_region
    else:
        return vehicle.location


def relocation_policy_shortest_wait(vehicle, current_time, simulator, **kwargs):
    """Relocate based on Shortest Wait policy (optimized)."""
    t_index = int(current_time) % simulator.lambda_.shape[0]
    R = simulator.lambda_.shape[1]
    N = simulator.N

    # Step 1: Precompute idle + relocating counts
    idle_relocating_counts = np.zeros(R)
    for v in simulator.vehicles:
        if v.status == IS_IDLE:
            idle_relocating_counts[v.location] += 1
        elif v.status == IS_RELOCATING and v.target_location is not None:
            idle_relocating_counts[v.target_location] += 1

    # Step 2: Compute expected wait times
    lambdas = simulator.lambda_[t_index]  # (R,)
    mus = simulator.mu_[t_index, vehicle.location]  # (R,)

    # Travel times for relocation (1 / mu_ij), avoid div by 0
    with np.errstate(divide='ignore', invalid='ignore'):
        travel_times = np.where(mus > 0, 1 / mus, np.inf)

    # Expected idle waiting time if arriving at each region
    expected_idle_waits = np.where(
        lambdas > 0,
        (idle_relocating_counts + 1) / (N * lambdas),
        np.inf
    )

    wait_times = travel_times + expected_idle_waits

    # For current region (staying), compute wait separately
    if lambdas[vehicle.location] > 0:
        wait_stay = idle_relocating_counts[vehicle.location] / (N * lambdas[vehicle.location])
    else:
        wait_stay = np.inf

    wait_times[vehicle.location] = np.inf  # Don't consider relocating to the same region

    # Step 3: Decision
    min_wait = np.min(wait_times)
    best_region = np.argmin(wait_times)

    if wait_stay <= min_wait:
        return vehicle.location
    else:
        return best_region
