import numpy as np
import matplotlib.pyplot as plt
from pyswarm import pso
import skfuzzy as fuzz
import skfuzzy.control as ctrl
import random

# Fuzzy System Creation Function (similar to original implementation)
def create_fuzzy_system(params):
    # Ensure parameters are sorted to satisfy a <= b <= c
    low_a, med_a, high_a = sorted(params[:3])
    low_b, med_b, high_b = sorted(params[3:6])

    # Inputs
    A_density = ctrl.Antecedent(np.arange(0, 101, 1), 'A_density')
    B_density = ctrl.Antecedent(np.arange(0, 101, 1), 'B_density')

    # Outputs
    A_duration = ctrl.Consequent(np.arange(10, 61, 1), 'A_duration')
    B_duration = ctrl.Consequent(np.arange(10, 61, 1), 'B_duration')

    # Membership functions for densities
    A_density['low'] = fuzz.trapmf(A_density.universe, [0, 0, low_a, med_a])
    A_density['medium'] = fuzz.trimf(A_density.universe, [low_a, med_a, high_a])
    A_density['high'] = fuzz.trapmf(A_density.universe, [med_a, high_a, 100, 100])

    B_density['low'] = fuzz.trapmf(B_density.universe, [0, 0, low_b, med_b])
    B_density['medium'] = fuzz.trimf(B_density.universe, [low_b, med_b, high_b])
    B_density['high'] = fuzz.trapmf(B_density.universe, [med_b, high_b, 100, 100])

    # Membership functions for durations
    A_duration['short'] = fuzz.trimf(A_duration.universe, [10, 20, 30])
    A_duration['medium'] = fuzz.trimf(A_duration.universe, [20, 40, 50])
    A_duration['long'] = fuzz.trimf(A_duration.universe, [40, 50, 60])

    B_duration['short'] = fuzz.trimf(B_duration.universe, [10, 20, 30])
    B_duration['medium'] = fuzz.trimf(B_duration.universe, [20, 40, 50])
    B_duration['long'] = fuzz.trimf(B_duration.universe, [40, 50, 60])

    # Define all 9 possible rules
    rules = [
        ctrl.Rule(A_density['low'] & B_density['low'], (A_duration['medium'], B_duration['medium'])),
        ctrl.Rule(A_density['low'] & B_density['medium'], (A_duration['short'], B_duration['long'])),
        ctrl.Rule(A_density['low'] & B_density['high'], (A_duration['short'], B_duration['long'])),
        ctrl.Rule(A_density['medium'] & B_density['low'], (A_duration['long'], B_duration['short'])),
        ctrl.Rule(A_density['medium'] & B_density['medium'], (A_duration['medium'], B_duration['medium'])),
        ctrl.Rule(A_density['medium'] & B_density['high'], (A_duration['short'], B_duration['long'])),
        ctrl.Rule(A_density['high'] & B_density['low'], (A_duration['long'], B_duration['short'])),
        ctrl.Rule(A_density['high'] & B_density['medium'], (A_duration['long'], B_duration['short'])),
        ctrl.Rule(A_density['high'] & B_density['high'], (A_duration['medium'], B_duration['medium']))
    ]

    # Control system
    system = ctrl.ControlSystem(rules)
    return ctrl.ControlSystemSimulation(system)

# Simulate Traffic Function
def simulate_traffic(params):
    fuzzy_system = create_fuzzy_system(params)

    # Example traffic patterns (simulated)
    traffic_data = [
        {"A_density": random.randint(0, 100), "B_density": random.randint(0, 100)},
        {"A_density": random.randint(0, 100), "B_density": random.randint(0, 100)},
        {"A_density": random.randint(0, 100), "B_density": random.randint(0, 100)},
    ]

    total_wait_time = 0
    total_vehicles = 0

    for data in traffic_data:
        # Ensure inputs are within valid range
        A_density_value = np.clip(data['A_density'], 0, 100)
        B_density_value = np.clip(data['B_density'], 0, 100)

        fuzzy_system.input['A_density'] = A_density_value
        fuzzy_system.input['B_density'] = B_density_value

        try:
            fuzzy_system.compute()
            A_duration = fuzzy_system.output['A_duration']
            B_duration = fuzzy_system.output['B_duration']

            # Simplistic metrics
            wait_time = (A_density_value * B_duration + B_density_value * A_duration) / (A_duration + B_duration)
            vehicles = A_density_value + B_density_value

            total_wait_time += wait_time
            total_vehicles += vehicles

        except KeyError:
            continue

    # Fitness: Higher vehicles passed per unit wait time
    return total_vehicles / total_wait_time if total_wait_time > 0 else 0

# Custom PSO with Convergence Tracking
def pso_with_convergence(fitness_func, lb, ub, swarmsize=20, maxiter=50):
    # Initialize tracking lists
    global_best_fitness_history = []
    iteration_history = []

    def tracking_fitness_func(params):
        fitness = fitness_func(params)
        global_best_fitness_history.append(fitness)
        iteration_history.append(len(global_best_fitness_history))
        return fitness

    # Run PSO with tracking
    best_params, best_score = pso(tracking_fitness_func, lb, ub, 
                                   swarmsize=swarmsize, 
                                   maxiter=maxiter)

    return best_params, best_score, global_best_fitness_history, iteration_history

# PSO Optimization with Visualization
def visualize_pso_convergence():
    # Parameter bounds for fuzzy membership functions
    lb = [0, 20, 40, 0, 20, 40]  # Lower bounds for low, medium, high (A and B)
    ub = [20, 60, 100, 20, 60, 100]  # Upper bounds for low, medium, high (A and B)

    # Run PSO with convergence tracking
    best_params, best_score, fitness_history, iteration_history = pso_with_convergence(
        simulate_traffic, lb, ub, swarmsize=20, maxiter=50
    )

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(iteration_history, fitness_history, 'b-', linewidth=2)
    plt.title('Particle Swarm Optimization - Convergence Curve', fontsize=15)
    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel('Best Fitness Score', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Highlight best point
    best_iteration_index = fitness_history.index(max(fitness_history))
    plt.scatter(iteration_history[best_iteration_index], 
                fitness_history[best_iteration_index], 
                color='red', s=200, label='Best Solution')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig('pso_convergence.png')
    plt.close()

    # Print optimization results
    print(f"Best Parameters: {best_params}")
    print(f"Best Fitness Score: {best_score}")

# Execute the visualization
visualize_pso_convergence()