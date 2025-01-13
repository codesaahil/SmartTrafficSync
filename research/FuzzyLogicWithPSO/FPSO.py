import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from pyswarm import pso
import random
from prettytable import PrettyTable

# --- Single Responsibility Principle (SRP) ---

# Separate class for Fuzzy System creation
class FuzzySystemFactory:
    @staticmethod
    def create(params):
        low_a, med_a, high_a = sorted(params[:3])
        low_b, med_b, high_b = sorted(params[3:6])

        a_density = ctrl.Antecedent(np.arange(0, 101, 1), 'A_density')
        b_density = ctrl.Antecedent(np.arange(0, 101, 1), 'B_density')

        a_duration = ctrl.Consequent(np.arange(10, 61, 1), 'A_duration')
        b_duration = ctrl.Consequent(np.arange(10, 61, 1), 'B_duration')

        a_density['low'] = fuzz.trapmf(a_density.universe, [0, 0, low_a, med_a])
        a_density['medium'] = fuzz.trimf(a_density.universe, [low_a, med_a, high_a])
        a_density['high'] = fuzz.trapmf(a_density.universe, [med_a, high_a, 100, 100])

        b_density['low'] = fuzz.trapmf(b_density.universe, [0, 0, low_b, med_b])
        b_density['medium'] = fuzz.trimf(b_density.universe, [low_b, med_b, high_b])
        b_density['high'] = fuzz.trapmf(b_density.universe, [med_b, high_b, 100, 100])

        a_duration['short'] = fuzz.trimf(a_duration.universe, [10, 20, 30])
        a_duration['medium'] = fuzz.trimf(a_duration.universe, [20, 40, 50])
        a_duration['long'] = fuzz.trimf(a_duration.universe, [40, 50, 60])

        b_duration['short'] = fuzz.trimf(b_duration.universe, [10, 20, 30])
        b_duration['medium'] = fuzz.trimf(b_duration.universe, [20, 40, 50])
        b_duration['long'] = fuzz.trimf(b_duration.universe, [40, 50, 60])

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

        system = ctrl.ControlSystem(rules)
        return ctrl.ControlSystemSimulation(system)

# Separate class for Traffic Simulation (Open/Closed Principle)
class TrafficSimulator:
    def simulate(self, controller, traffic_data):
        return controller.control_traffic(traffic_data)

# Abstract base class for Traffic Controllers (Liskov Substitution Principle, Interface Segregation Principle)
class TrafficController:
    def control_traffic(self, traffic_data):
        raise NotImplementedError

class FuzzyTrafficController(TrafficController):
    def __init__(self, params):
        self.fuzzy_system = FuzzySystemFactory.create(params)

    def control_traffic(self, traffic_data):
        a_density_value = np.clip(traffic_data['A_density'], 0, 100)
        b_density_value = np.clip(traffic_data['B_density'], 0, 100)

        self.fuzzy_system.input['A_density'] = a_density_value
        self.fuzzy_system.input['B_density'] = b_density_value
        try:
            self.fuzzy_system.compute()
            return self.fuzzy_system.output['A_duration'], self.fuzzy_system.output['B_duration']
        except KeyError as e:
            print(f"KeyError during fuzzy computation: {e}")
            print(f"Inputs: A_density={a_density_value}, B_density={b_density_value}")
            return None, None

class FixedTimeTrafficController(TrafficController):
    def __init__(self, fixed_duration_a, fixed_duration_b):
        self.fixed_duration_a = fixed_duration_a
        self.fixed_duration_b = fixed_duration_b

    def control_traffic(self, traffic_data):
        return self.fixed_duration_a, self.fixed_duration_b

# Separate class for Metric Calculation (SRP)
class MetricsCalculator:
    def calculate_metrics(self, a_duration, b_duration, traffic_data):
        total_wait_time_a = 0
        total_wait_time_b = 0
        total_vehicles_a_served = 0
        total_vehicles_b_served = 0
        epsilon = 1e-9
        all_wait_times_a = []
        all_wait_times_b = []

        a_density_value = np.clip(traffic_data['A_density'], 0, 100)
        b_density_value = np.clip(traffic_data['B_density'], 0, 100)

        if a_duration is None or b_duration is None:
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
                "average_system_time": 1e9,
            }

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

# --- Dependency Inversion Principle (DIP) ---

# Abstract Fitness Evaluator
class FitnessEvaluator:
    def evaluate(self, controller, traffic_data):
        raise NotImplementedError

class FuzzyFitnessEvaluator(FitnessEvaluator):
    def __init__(self, metrics_calculator, weight_wait=1.0, weight_throughput=1.0):
        self.metrics_calculator = metrics_calculator
        self.weight_wait = weight_wait
        self.weight_throughput = weight_throughput

    def evaluate(self, controller, traffic_data):
        a_duration, b_duration = controller.control_traffic(traffic_data)
        metrics = self.metrics_calculator.calculate_metrics(a_duration, b_duration, traffic_data)
        average_system_time = metrics.get("average_system_time", 1e9)
        average_throughput = metrics.get("average_throughput", 0)
        fitness = self.weight_wait * average_system_time - self.weight_throughput * average_throughput
        return fitness

# --- Helper functions ---
def calculate_average_metrics(simulator, controller, num_simulations=100):
    all_metrics = []
    metrics_calculator = MetricsCalculator()
    for _ in range(num_simulations):
        traffic_data = {"A_density": random.randint(0, 100), "B_density": random.randint(0, 100)}
        a_duration, b_duration = simulator.simulate(controller, traffic_data)
        metrics = metrics_calculator.calculate_metrics(a_duration, b_duration, traffic_data)
        all_metrics.append(metrics)

    average_metrics = {}
    if all_metrics:
        for key in all_metrics[0].keys():
            average_metrics[key] = np.mean([m[key] for m in all_metrics])
    return average_metrics

# --- Optimization ---
def optimize_fuzzy_system(fitness_evaluator, lb, ub, swarmsize=20, maxiter=50):
    def fitness_function_for_pso(params):
        controller = FuzzyTrafficController(params)
        traffic_data = {"A_density": random.randint(0, 100), "B_density": random.randint(0, 100)}
        return fitness_evaluator.evaluate(controller, traffic_data)

    best_params, best_score = pso(fitness_function_for_pso, lb, ub, swarmsize=swarmsize, maxiter=maxiter)
    return best_params, best_score

# --- Main Execution ---
if __name__ == "__main__":
    # Optimization
    weight_wait = 1.0
    weight_throughput = 1.0
    lb = [0, 20, 40, 0, 20, 40]
    ub = [20, 60, 100, 20, 60, 100]

    metrics_calculator = MetricsCalculator()
    fitness_evaluator = FuzzyFitnessEvaluator(metrics_calculator, weight_wait, weight_throughput)
    best_params, best_score = optimize_fuzzy_system(fitness_evaluator, lb, ub)
    print(f"Best Parameters (Fuzzy): {best_params}")
    print(f"Best Fitness Score (Fuzzy): {best_score:.2f}")

    # Simulation and Comparison
    num_simulations = 100
    simulator = TrafficSimulator()

    # Fuzzy Controller
    optimized_fuzzy_controller = FuzzyTrafficController(best_params)
    average_fuzzy_metrics = calculate_average_metrics(simulator, optimized_fuzzy_controller, num_simulations)

    # Fixed Time Controller
    fixed_time_controller = FixedTimeTrafficController(30, 30)
    average_fixed_time_metrics = calculate_average_metrics(simulator, fixed_time_controller, num_simulations)

    # Comparison Table
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

    # Overall Performance
    print("\nOverall Performance (Averaged):")
    avg_system_time_fuzzy = average_fuzzy_metrics.get('average_system_time', float('inf'))
    avg_system_time_fixed = average_fixed_time_metrics.get('average_system_time', float('inf'))

    avg_throughput_fuzzy = average_fuzzy_metrics.get('average_throughput', 0)
    avg_throughput_fixed = average_fixed_time_metrics.get('average_throughput', 0)

    threshold = 0.01

    if avg_system_time_fuzzy < (avg_system_time_fixed - threshold) and avg_throughput_fuzzy > (avg_throughput_fixed + threshold):
        print("The Fuzzy Logic Controller performs better than the Fixed Time Controller based on average system time and throughput over multiple simulations.")
    elif avg_system_time_fixed < (avg_system_time_fuzzy - threshold) and avg_throughput_fixed > (avg_throughput_fuzzy + threshold):
        print("The Fixed Time Controller performs better than the Fuzzy Logic Controller based on average system time and throughput over multiple simulations.")
    else:
        print("The performance difference between the Fuzzy Logic Controller and the Fixed Time Controller is not conclusive based on average system time and throughput over multiple simulations.")