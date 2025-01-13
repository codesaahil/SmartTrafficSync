import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from pyswarm import pso
import random
import time
from prettytable import PrettyTable

# Create Fuzzy Logic System
def create_fuzzy_system(params):
    # Ensure parameters are sorted to satisfy a <= b <= c
    low_a, med_a, high_a = sorted(params[:3])
    low_b, med_b, high_b = sorted(params[3:6])

    # Inputs
    a_density = ctrl.Antecedent(np.arange(0, 101, 1), 'A_density')
    b_density = ctrl.Antecedent(np.arange(0, 101, 1), 'B_density')

    # Outputs
    a_duration = ctrl.Consequent(np.arange(10, 61, 1), 'A_duration')
    b_duration = ctrl.Consequent(np.arange(10, 61, 1), 'B_duration')

    # Membership functions for densities
    a_density['low'] = fuzz.trapmf(a_density.universe, [0, 0, low_a, med_a])
    a_density['medium'] = fuzz.trimf(a_density.universe, [low_a, med_a, high_a])
    a_density['high'] = fuzz.trapmf(a_density.universe, [med_a, high_a, 100, 100])

    b_density['low'] = fuzz.trapmf(b_density.universe, [0, 0, low_b, med_b])
    b_density['medium'] = fuzz.trimf(b_density.universe, [low_b, med_b, high_b])
    b_density['high'] = fuzz.trapmf(b_density.universe, [med_b, high_b, 100, 100])

    # Membership functions for durations
    a_duration['short'] = fuzz.trimf(a_duration.universe, [10, 20, 30])
    a_duration['medium'] = fuzz.trimf(a_duration.universe, [20, 40, 50])
    a_duration['long'] = fuzz.trimf(a_duration.universe, [40, 50, 60])

    b_duration['short'] = fuzz.trimf(b_duration.universe, [10, 20, 30])
    b_duration['medium'] = fuzz.trimf(b_duration.universe, [20, 40, 50])
    b_duration['long'] = fuzz.trimf(b_duration.universe, [40, 50, 60])

    # Define all 9 possible rules
    rules = [
        ctrl.Rule(a_density['low'] & b_density['low'], (a_duration['medium'], b_duration['medium'])),
        ctrl.Rule(a_density['low'] & b_density['medium'], (a_duration['short'], b_duration['long'])),
        ctrl.Rule(a_density['low'] & b_density['high'], (a_duration['short'], b_duration['long'])),
        ctrl.Rule(a_density['medium'] & b_density['low'], (a_duration['long'], b_duration['short'])),
        ctrl.Rule(a_density['medium'] & b_density['medium'], (a_duration['medium'], b_duration['medium'])),
        ctrl.Rule(a_density['medium'] & b_density['high'], (a_duration['short'], b_duration['long'])),
        ctrl.Rule(a_density['high'] & b_density['low'], (a_duration['long'], b_duration['short'])),
        ctrl.Rule(a_density['high'] & b_density['medium'], (a_duration['long'], b_duration['short'])),
        ctrl.Rule(a_density['high'] & b_density['high'], (a_duration['medium'], b_duration['medium']))
    ]

    # Control system
    system = ctrl.ControlSystem(rules)
    return ctrl.ControlSystemSimulation(system)

# Simulate Traffic with Current Parameters
def simulate_traffic(params):
    fuzzy_system = create_fuzzy_system(params)

    traffic_data = {"A_density": random.randint(0, 100), "B_density": random.randint(0, 100)}

    a_density_value = np.clip(traffic_data['A_density'], 0, 100)
    b_density_value = np.clip(traffic_data['B_density'], 0, 100)

    fuzzy_system.input['A_density'] = a_density_value
    fuzzy_system.input['B_density'] = b_density_value

    try:
        fuzzy_system.compute()
        metrics = calculate_metrics_single_run(fuzzy_system, traffic_data)
        return metrics

    except KeyError as e:
        print(f"KeyError during fuzzy computation: {e}")
        print(f"Inputs: A_density={a_density_value}, B_density={b_density_value}")
        # Return a large but finite penalty, and 0 for throughput
        return {"average_system_time": 1e9, "average_throughput": 0}

def calculate_metrics_single_run(fuzzy_system, traffic_data):
    total_wait_time_a = 0
    total_wait_time_b = 0
    total_vehicles_a_served = 0
    total_vehicles_b_served = 0
    epsilon = 1e-9
    all_wait_times_a = []
    all_wait_times_b = []

    a_density_value = np.clip(traffic_data['A_density'], 0, 100)
    b_density_value = np.clip(traffic_data['B_density'], 0, 100)

    fuzzy_system.input['A_density'] = a_density_value
    fuzzy_system.input['B_density'] = b_density_value

    try:
        fuzzy_system.compute()
        a_duration = fuzzy_system.output['A_duration']
        b_duration = fuzzy_system.output['B_duration']

        # The time a vehicle in A "waits" is the time B's light is green (B_duration)
        avg_wait_per_vehicle_a = b_duration
        avg_wait_per_vehicle_b = a_duration
        all_wait_times_a.extend([avg_wait_per_vehicle_a] * int(a_density_value))
        all_wait_times_b.extend([avg_wait_per_vehicle_b] * int(b_density_value))

        total_wait_time_a += a_density_value * b_duration
        total_wait_time_b += b_density_value * a_duration

        vehicles_a_served = min(a_density_value, (a_duration / 60) * 100)
        vehicles_b_served = min(b_density_value, (b_duration / 60) * 100)

        total_vehicles_a_served += vehicles_a_served
        total_vehicles_b_served += vehicles_b_served

    except KeyError as e:
        print(f"KeyError during fuzzy computation in metrics: {e}")
        print(f"Inputs: A_density={a_density_value}, B_density={b_density_value}")
        return {
            "average_wait_time_A": 1e9,
            "average_wait_time_B": 1e9,
            "total_wait_time_A": 1e9,
            "total_wait_time_B": 1e9,
            "total_wait_time": 2e9,
            "total_vehicles_A_served": 0,
            "total_vehicles_B_served": 0,
            "total_vehicles_served": 0,
            "throughput_A": 0,
            "throughput_B": 0,
            "average_throughput": 0,
            "wait_time_disparity": 0,
        }

    avg_wait_time_a = np.mean(all_wait_times_a) if all_wait_times_a else 0
    avg_wait_time_b = np.mean(all_wait_times_b) if all_wait_times_b else 0
    total_vehicles_served = total_vehicles_a_served + total_vehicles_b_served

    metrics = {
        "average_wait_time_A": avg_wait_time_a,  # Average time a vehicle in A waits
        "average_wait_time_B": avg_wait_time_b,  # Average time a vehicle in B waits
        "total_wait_time_A": total_wait_time_a,
        "total_wait_time_B": total_wait_time_b,
        "total_wait_time": total_wait_time_a + total_wait_time_b,
        "total_vehicles_A_served": total_vehicles_a_served,
        "total_vehicles_B_served": total_vehicles_b_served,
        "total_vehicles_served": total_vehicles_served,
        "throughput_A": total_vehicles_a_served / (a_density_value + epsilon) if a_density_value > 0 else 0,
        "throughput_B": total_vehicles_b_served / (b_density_value + epsilon) if b_density_value > 0 else 0,
        "average_throughput": total_vehicles_served / (a_density_value + b_density_value + epsilon) if (a_density_value + b_density_value) > 0 else 0,
        "wait_time_disparity": abs(avg_wait_time_a - avg_wait_time_b),
        "average_system_time": (total_wait_time_a + total_wait_time_b) / (total_vehicles_served + epsilon) if total_vehicles_served > 0 else 0, # Total wait time per served vehicle
    }

    return metrics

# Fitness Function for PSO (Minimize)
def fitness_function(params, weight_wait=1.0, weight_throughput=1.0):
    metrics = simulate_traffic(params)
    average_system_time = metrics.get("average_system_time", 1e9)
    average_throughput = metrics.get("average_throughput", 0)

    fitness = weight_wait * average_system_time - weight_throughput * average_throughput
    return fitness

# Function to calculate performance metrics with multiple runs and averaging
def calculate_average_metrics(params, num_simulations=100):
    fuzzy_system = create_fuzzy_system(params)
    all_metrics = []

    for _ in range(num_simulations):
        traffic_data = {"A_density": random.randint(0, 100), "B_density": random.randint(0, 100)}
        metrics = calculate_metrics_single_run(fuzzy_system, traffic_data)
        all_metrics.append(metrics)

    average_metrics = {}
    if all_metrics:
        for key in all_metrics[0].keys():
            average_metrics[key] = np.mean([m[key] for m in all_metrics])
    return average_metrics

# Simulate Fixed Time Traffic Light with multiple runs and averaging
def simulate_average_fixed_time_traffic(fixed_duration_a, fixed_duration_b, num_simulations=100):
    all_metrics = []
    for _ in range(num_simulations):
        traffic_data = {"A_density": random.randint(0, 100), "B_density": random.randint(0, 100)}
        metrics = simulate_fixed_time_traffic_single_run(fixed_duration_a, fixed_duration_b, traffic_data)
        all_metrics.append(metrics)

    average_metrics = {}
    if all_metrics:
        for key in all_metrics[0].keys():
            average_metrics[key] = np.mean([m[key] for m in all_metrics])
    return average_metrics

def simulate_fixed_time_traffic_single_run(fixed_duration_a, fixed_duration_b, traffic_data):
    total_wait_time_a = 0
    total_wait_time_b = 0
    total_vehicles_a_served = 0
    total_vehicles_b_served = 0
    epsilon = 1e-9
    all_wait_times_a = []
    all_wait_times_b = []

    a_density_value = np.clip(traffic_data['A_density'], 0, 100)
    b_density_value = np.clip(traffic_data['B_density'], 0, 100)

    avg_wait_per_vehicle_a = fixed_duration_b
    avg_wait_per_vehicle_b = fixed_duration_a
    all_wait_times_a.extend([avg_wait_per_vehicle_a] * int(a_density_value))
    all_wait_times_b.extend([avg_wait_per_vehicle_b] * int(b_density_value))

    total_wait_time_a += a_density_value * fixed_duration_b
    total_wait_time_b += b_density_value * fixed_duration_a

    vehicles_a_served = min(a_density_value, (fixed_duration_a / 60) * 100)
    vehicles_b_served = min(b_density_value, (fixed_duration_b / 60) * 100)

    total_vehicles_a_served += vehicles_a_served
    total_vehicles_b_served += vehicles_b_served

    avg_wait_time_a = np.mean(all_wait_times_a) if all_wait_times_a else 0
    avg_wait_time_b = np.mean(all_wait_times_b) if all_wait_times_b else 0
    total_vehicles_served = total_vehicles_a_served + total_vehicles_b_served

    metrics = {
        "average_wait_time_A": avg_wait_time_a,
        "average_wait_time_B": avg_wait_time_b,
        "total_wait_time_A": total_wait_time_a,
        "total_wait_time_B": total_wait_time_b,
        "total_wait_time": total_wait_time_a + total_wait_time_b,
        "total_vehicles_A_served": total_vehicles_a_served,
        "total_vehicles_B_served": total_vehicles_b_served,
        "total_vehicles_served": total_vehicles_served,
        "throughput_A": total_vehicles_a_served / (a_density_value + epsilon) if a_density_value > 0 else 0,
        "throughput_B": total_vehicles_b_served / (b_density_value + epsilon) if b_density_value > 0 else 0,
        "average_throughput": total_vehicles_served / (a_density_value + b_density_value + epsilon) if (a_density_value + b_density_value) > 0 else 0,
        "wait_time_disparity": abs(avg_wait_time_a - avg_wait_time_b),
        "average_system_time": (total_wait_time_a + total_wait_time_b) / (total_vehicles_served + epsilon) if total_vehicles_served > 0 else 0,
    }
    return metrics

# PSO Optimization
def optimize_fuzzy_system(weight_wait=1.0, weight_throughput=1.0):
    lb = [0, 20, 40, 0, 20, 40]
    ub = [20, 60, 100, 20, 60, 100]
    fitness_func_with_weights = lambda params: fitness_function(params, weight_wait, weight_throughput)
    best_params, best_score = pso(fitness_func_with_weights, lb, ub, swarmsize=20, maxiter=50)
    return best_params, best_score

# Run Optimization with weights for wait time and throughput
weight_wait = 1.0
weight_throughput = 1.0
best_params, best_score = optimize_fuzzy_system(weight_wait, weight_throughput)
print(f"Best Parameters (Fuzzy): {best_params}")
print(f"Best Fitness Score (Fuzzy): {best_score:.2f}")

# Number of simulations for calculating average metrics
num_simulations = 100

# Calculate average metrics for the optimized fuzzy system
average_fuzzy_metrics = calculate_average_metrics(best_params, num_simulations)

# Simulate fixed time traffic light and calculate average metrics
fixed_duration_A = 30
fixed_duration_B = 30
average_fixed_time_metrics = simulate_average_fixed_time_traffic(fixed_duration_A, fixed_duration_B, num_simulations)

# Compare the average metrics in a table
table = PrettyTable()
table.field_names = ["Metric", "Fuzzy Logic Controller (Avg)", "Fixed Time Controller (Avg)", "Better"]

for metric in average_fuzzy_metrics.keys():
    fuzzy_value = average_fuzzy_metrics[metric]
    fixed_value = average_fixed_time_metrics.get(metric, None)
    better = ""

    if fixed_value is not None:
        if ("wait_time" in metric and "disparity" not in metric) or "disparity" in metric or "system_time" in metric:
            if fuzzy_value < fixed_value:
                better = "Fuzzy"
            elif fixed_value < fuzzy_value:
                better = "Fixed"
        elif "throughput" in metric or "served" in metric:
            if fuzzy_value > fixed_value:
                better = "Fuzzy"
            elif fixed_value > fuzzy_value:
                better = "Fixed"

        table.add_row([metric.replace("_", " ").title(), f"{fuzzy_value:.2f}", f"{fixed_value:.2f}", better])
    else:
        table.add_row([metric.replace("_", " ").title(), f"{fuzzy_value:.2f}", "N/A", ""])

print(f"\nPerformance Comparison (Averaged over {num_simulations} simulations):")
print(table)

# Determine which is better overall (simple comparison based on a few key average metrics)
print("\nOverall Performance (Averaged):")
avg_system_time_fuzzy = average_fuzzy_metrics.get('average_system_time', float('inf'))
avg_system_time_fixed = average_fixed_time_metrics.get('average_system_time', float('inf'))

avg_throughput_fuzzy = average_fuzzy_metrics.get('average_throughput', 0)
avg_throughput_fixed = average_fixed_time_metrics.get('average_throughput', 0)

if avg_system_time_fuzzy < avg_system_time_fixed and avg_throughput_fuzzy > avg_throughput_fixed:
    print("The Fuzzy Logic Controller performs better than the Fixed Time Controller based on average system time and throughput over multiple simulations.")
elif avg_system_time_fixed < avg_system_time_fuzzy and avg_throughput_fixed > avg_throughput_fuzzy:
    print("The Fixed Time Controller performs better than the Fuzzy Logic Controller based on average system time and throughput over multiple simulations.")
else:
    print("The performance difference between the Fuzzy Logic Controller and the Fixed Time Controller is not conclusive based on average system time and throughput over multiple simulations.")