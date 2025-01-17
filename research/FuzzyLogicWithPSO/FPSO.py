import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from pyswarm import pso
import random
import os
import dataframe_image as dfi
from abc import ABC, abstractmethod

class FuzzyMembershipFunctionFactory:
    """A factory class for creating fuzzy membership functions."""
    @staticmethod
    def create_trapmf(universe, params):
        """Creates a trapezoidal membership function.

        Args:
            universe (np.ndarray): The universe of discourse.
            params (list): Parameters defining the trapezoid [a, b, c, d].

        Returns:
            np.ndarray: The trapezoidal membership function.
        """
        return fuzz.trapmf(universe, params)

    @staticmethod
    def create_trimf(universe, params):
        """Creates a triangular membership function.

        Args:
            universe (np.ndarray): The universe of discourse.
            params (list): Parameters defining the triangle [a, b, c].

        Returns:
            np.ndarray: The triangular membership function.
        """
        return fuzz.trimf(universe, params)

class FuzzyRuleBuilder:
    """A utility class for building fuzzy rules."""
    @staticmethod
    def create_rule(antecedent, consequent):
        """Creates a fuzzy control rule.

        Args:
            antecedent (skfuzzy.control.term.Term): The antecedent of the rule.
            consequent (tuple): The consequent(s) of the rule.

        Returns:
            skfuzzy.control.Rule: The created fuzzy rule.
        """
        return ctrl.Rule(antecedent, consequent)

class FuzzySystemBuilder:
    """A builder class for constructing a fuzzy control system."""
    def __init__(self):
        """Initializes the FuzzySystemBuilder."""
        self.antecedents = {}
        self.consequents = {}
        self.rules = []

    def add_antecedent(self, name, universe):
        """Adds an antecedent variable to the fuzzy system.

        Args:
            name (str): The name of the antecedent variable.
            universe (np.ndarray): The universe of discourse for the antecedent.

        Returns:
            skfuzzy.control.Antecedent: The created antecedent.
        """
        self.antecedents[name] = ctrl.Antecedent(universe, name)
        return self.antecedents[name]

    def add_consequent(self, name, universe):
        """Adds a consequent variable to the fuzzy system.

        Args:
            name (str): The name of the consequent variable.
            universe (np.ndarray): The universe of discourse for the consequent.

        Returns:
            skfuzzy.control.Consequent: The created consequent.
        """
        self.consequents[name] = ctrl.Consequent(universe, name)
        return self.consequents[name]

    def add_rule(self, rule):
        """Adds a fuzzy rule to the system.

        Args:
            rule (skfuzzy.control.Rule): The fuzzy rule to add.
        """
        self.rules.append(rule)

    def build(self):
        """Builds the fuzzy control system.

        Returns:
            skfuzzy.control.ControlSystemSimulation: The simulation object for the built fuzzy system.
        """
        system = ctrl.ControlSystem(self.rules)
        return ctrl.ControlSystemSimulation(system)

class IFuzzySystemFactory(ABC):
    """An abstract base class defining the interface for fuzzy system factories."""
    @abstractmethod
    def create(self, params):
        """Creates a fuzzy control system.

        Args:
            params (list): Parameters to configure the fuzzy system.

        Returns:
            skfuzzy.control.ControlSystemSimulation: The simulation object for the created fuzzy system.
        """
        pass

class ParameterizedFuzzySystemFactory(IFuzzySystemFactory):
    """A concrete factory for creating parameterized fuzzy control systems."""
    def create(self, params):
        """Creates a fuzzy control system based on the provided parameters.

        Args:
            params (list): A list of 6 parameters defining the membership functions for traffic densities.

        Returns:
            skfuzzy.control.ControlSystemSimulation: The simulation object for the created fuzzy system.
        """
        low_a, med_a, high_a = sorted(params[:3])
        low_b, med_b, high_b = sorted(params[3:6])

        system_builder = FuzzySystemBuilder()

        a_density = system_builder.add_antecedent('A_density', np.arange(0, 101, 1))
        b_density = system_builder.add_antecedent('B_density', np.arange(0, 101, 1))

        a_duration = system_builder.add_consequent('A_duration', np.arange(10, 61, 1))
        b_duration = system_builder.add_consequent('B_duration', np.arange(10, 61, 1))

        a_density['low'] = FuzzyMembershipFunctionFactory.create_trapmf(a_density.universe, [0, 0, low_a, med_a])
        a_density['medium'] = FuzzyMembershipFunctionFactory.create_trimf(a_density.universe, [low_a, med_a, high_a])
        a_density['high'] = FuzzyMembershipFunctionFactory.create_trapmf(a_density.universe, [med_a, high_a, 100, 100])

        b_density['low'] = FuzzyMembershipFunctionFactory.create_trapmf(b_density.universe, [0, 0, low_b, med_b])
        b_density['medium'] = FuzzyMembershipFunctionFactory.create_trimf(b_density.universe, [low_b, med_b, high_b])
        b_density['high'] = FuzzyMembershipFunctionFactory.create_trapmf(b_density.universe, [med_b, high_b, 100, 100])

        a_duration['short'] = FuzzyMembershipFunctionFactory.create_trimf(a_duration.universe, [10, 20, 30])
        a_duration['medium'] = FuzzyMembershipFunctionFactory.create_trimf(a_duration.universe, [20, 40, 50])
        a_duration['long'] = FuzzyMembershipFunctionFactory.create_trimf(a_duration.universe, [40, 50, 60])

        b_duration['short'] = FuzzyMembershipFunctionFactory.create_trimf(b_duration.universe, [10, 20, 30])
        b_duration['medium'] = FuzzyMembershipFunctionFactory.create_trimf(b_duration.universe, [20, 40, 50])
        b_duration['long'] = FuzzyMembershipFunctionFactory.create_trimf(b_duration.universe, [40, 50, 60])

        rules = [
            FuzzyRuleBuilder.create_rule(a_density['low'] & b_density['low'], (a_duration['medium'], b_duration['medium'])),
            FuzzyRuleBuilder.create_rule(a_density['low'] & b_density['medium'], (a_duration['short'], b_duration['long'])),
            FuzzyRuleBuilder.create_rule(a_density['low'] & b_density['high'], (a_duration['short'], b_duration['long'])),
            FuzzyRuleBuilder.create_rule(a_density['medium'] & b_density['low'], (a_duration['long'], b_duration['short'])),
            FuzzyRuleBuilder.create_rule(a_density['medium'] & b_density['medium'], (a_duration['medium'], b_duration['medium'])),
            FuzzyRuleBuilder.create_rule(a_density['medium'] & b_density['high'], (a_duration['short'], b_duration['long'])),
            FuzzyRuleBuilder.create_rule(a_density['high'] & b_density['low'], (a_duration['long'], b_duration['short'])),
            FuzzyRuleBuilder.create_rule(a_density['high'] & b_density['medium'], (a_duration['long'], b_duration['short'])),
            FuzzyRuleBuilder.create_rule(a_density['high'] & b_density['high'], (a_duration['medium'], b_duration['medium']))
        ]

        for rule in rules:
            system_builder.add_rule(rule)

        return system_builder.build()

class TrafficController(ABC):
    """An abstract base class for traffic controllers."""
    @abstractmethod
    def control_traffic(self, traffic_data):
        """Abstract method to control traffic signals.

        Args:
            traffic_data (dict): A dictionary containing traffic density data.
        """
        pass

class FuzzyTrafficController(TrafficController):
    """A traffic controller using a fuzzy logic system."""
    def __init__(self, fuzzy_system_factory: IFuzzySystemFactory, params):
        """Initializes the FuzzyTrafficController.

        Args:
            fuzzy_system_factory (IFuzzySystemFactory): The factory to create the fuzzy system.
            params (list): Parameters for the fuzzy system.
        """
        self.fuzzy_system = fuzzy_system_factory.create(params)

    def control_traffic(self, traffic_data):
        """Controls traffic signals using the fuzzy logic system.

        Args:
            traffic_data (dict): A dictionary containing 'A_density' and 'B_density'.

        Returns:
            tuple: The durations for traffic signals A and B, or (None, None) if an error occurs.
        """
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
    """A traffic controller with fixed signal durations."""
    def __init__(self, fixed_duration_a, fixed_duration_b):
        """Initializes the FixedTimeTrafficController.

        Args:
            fixed_duration_a (int): The fixed duration for traffic signal A.
            fixed_duration_b (int): The fixed duration for traffic signal B.
        """
        self.fixed_duration_a = fixed_duration_a
        self.fixed_duration_b = fixed_duration_b

    def control_traffic(self, traffic_data):
        """Returns the pre-set fixed durations for traffic signals.

        Args:
            traffic_data (dict): The traffic data (ignored by this controller).

        Returns:
            tuple: The fixed durations for traffic signals A and B.
        """
        return self.fixed_duration_a, self.fixed_duration_b

class TrafficSimulator:
    """Simulates traffic flow using a given traffic controller."""
    def __init__(self):
        """Initializes the TrafficSimulator."""
        pass

    def simulate(self, controller: TrafficController, traffic_data):
        """Simulates one time step of traffic flow.

        Args:
            controller (TrafficController): The traffic controller to use.
            traffic_data (dict): The current traffic density data.

        Returns:
            tuple: The durations for traffic signals A and B determined by the controller.
        """
        return controller.control_traffic(traffic_data)

class MetricsCalculator:
    """Calculates performance metrics for a traffic control system."""
    def calculate_metrics(self, a_duration, b_duration, traffic_data):
        """Calculates various performance metrics.

        Args:
            a_duration (float): The duration of traffic signal A.
            b_duration (float): The duration of traffic signal B.
            traffic_data (dict): The traffic density data.

        Returns:
            dict: A dictionary of calculated performance metrics.
        """
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

class FitnessEvaluator(ABC):
    """An abstract base class for fitness evaluators."""
    @abstractmethod
    def evaluate(self, controller: TrafficController, traffic_data):
        """Abstract method to evaluate the fitness of a traffic controller.

        Args:
            controller (TrafficController): The traffic controller to evaluate.
            traffic_data (dict): The traffic density data.
        """
        pass

class FuzzyFitnessEvaluator(FitnessEvaluator):
    """Evaluates the fitness of a fuzzy traffic controller."""
    def __init__(self, metrics_calculator: MetricsCalculator, weight_wait=1.0, weight_throughput=1.0):
        """Initializes the FuzzyFitnessEvaluator.

        Args:
            metrics_calculator (MetricsCalculator): The calculator for performance metrics.
            weight_wait (float): Weight for average system time in fitness calculation.
            weight_throughput (float): Weight for average throughput in fitness calculation.
        """
        self.metrics_calculator = metrics_calculator
        self.weight_wait = weight_wait
        self.weight_throughput = weight_throughput

    def evaluate(self, controller: FuzzyTrafficController, traffic_data):
        """Evaluates the fitness of a fuzzy traffic controller.

        Args:
            controller (FuzzyTrafficController): The fuzzy traffic controller to evaluate.
            traffic_data (dict): The traffic density data.

        Returns:
            float: The fitness score of the controller.
        """
        a_duration, b_duration = controller.control_traffic(traffic_data)
        metrics = self.metrics_calculator.calculate_metrics(a_duration, b_duration, traffic_data)
        average_system_time = metrics.get("average_system_time", 1e9)
        average_throughput = metrics.get("average_throughput", 0)
        fitness = (self.weight_wait * average_system_time) - (self.weight_throughput * average_throughput)
        return fitness

def optimize_fuzzy_system(fitness_evaluator: FuzzyFitnessEvaluator, lb, ub, swarmsize=20, maxiter=50):
    """Optimizes the parameters of the fuzzy system using Particle Swarm Optimization.

    Args:
        fitness_evaluator (FuzzyFitnessEvaluator): The evaluator to determine the fitness of parameters.
        lb (list): Lower bounds for the parameters.
        ub (list): Upper bounds for the parameters.
        swarmsize (int): The number of particles in the swarm.
        maxiter (int): The maximum number of iterations for the optimization.

    Returns:
        tuple: The best parameters and the best fitness score found, and the fitness history.
    """
    history_fitness = []
    def fitness_function_for_pso(params):
        fuzzy_system_factory = ParameterizedFuzzySystemFactory()
        controller = FuzzyTrafficController(fuzzy_system_factory, params)
        traffic_data = {"A_density": random.randint(0, 100), "B_density": random.randint(0, 100)}
        fitness_value = fitness_evaluator.evaluate(controller, traffic_data)
        history_fitness.append(fitness_value)
        return fitness_value

    best_params, best_score = pso(fitness_function_for_pso, lb, ub, swarmsize=swarmsize, maxiter=maxiter)
    return best_params, best_score, history_fitness

def plot_pso_convergence(fitness_history, title="PSO Convergence"):
    """Plots the convergence graph of the PSO algorithm.

    Args:
        fitness_history (list): A list of fitness values recorded during PSO.
        title (str): The title of the plot.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(fitness_history)
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Fitness Value")
    plt.grid(True)
    plt.savefig(os.path.join('graphs', f'pso_convergence_{title.lower().replace(" ", "_")}.png'))
    plt.close()

def run_simulations(simulator: TrafficSimulator, controller: TrafficController, traffic_data_list):
    """Runs multiple simulations with a given controller and traffic data.

    Args:
        simulator (TrafficSimulator): The simulator to use.
        controller (TrafficController): The traffic controller to test.
        traffic_data_list (list): A list of traffic data dictionaries.

    Returns:
        pandas.DataFrame: A DataFrame containing the performance metrics for each simulation.
    """
    all_metrics = []
    metrics_calculator = MetricsCalculator()
    for traffic_data in traffic_data_list:
        a_duration, b_duration = simulator.simulate(controller, traffic_data)
        metrics = metrics_calculator.calculate_metrics(a_duration, b_duration, traffic_data)
        all_metrics.append(metrics)
    return pd.DataFrame(all_metrics)

def save_table_as_png(df, filename, folder='tables'):
    """Saves a pandas DataFrame as a PNG image.

    Args:
        df (pandas.DataFrame): The DataFrame to save.
        filename (str): The name of the file to save as.
        folder (str): The folder to save the file in.
    """
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)
    dfi.export(df, filepath)

if __name__ == "__main__":
    # --- Optimization and Initial Setup ---
    weight_wait = 1.0
    weight_throughput = 1.0
    lb = [0, 20, 40, 0, 20, 40]
    ub = [20, 60, 100, 20, 60, 100]

    metrics_calculator = MetricsCalculator()
    fitness_evaluator = FuzzyFitnessEvaluator(metrics_calculator, weight_wait, weight_throughput)

    simulator = TrafficSimulator()

    num_simulations = 1000
    traffic_data_list = [{"A_density": random.randint(0, 100), "B_density": random.randint(0, 100)} for _ in range(num_simulations)]

    # --- Optimization Phase: Parameter Tuning with Increased Iterations and Swarm Size ---
    print("\n--- Optimization Phase: Parameter Tuning with Increased Iterations and Swarm Size ---")
    best_params_high_iter, best_score_high_iter, history_high_iter = optimize_fuzzy_system(fitness_evaluator, lb, ub, swarmsize=40, maxiter=1000)
    print(f"Best Parameters (High Iter/Swarm): {best_params_high_iter}")
    print(f"Best Fitness Score (High Iter/Swarm): {best_score_high_iter:.2f}")
    plot_pso_convergence(history_high_iter, title="PSO Convergence (Less is better)")

    # --- Phase: Multiple Independent Optimization Runs for Robustness ---
    num_optimization_runs = 3
    all_optimized_results = []
    print("\n--- Phase: Multiple Independent Optimization Runs for Robustness ---")
    for i in range(num_optimization_runs):
        print(f"\n--- Optimization Run {i+1}/{num_optimization_runs} ---")
        best_params_run, best_score_run, history_run = optimize_fuzzy_system(fitness_evaluator, lb, ub)
        print(f"Best Parameters (Run {i+1}): {best_params_run}")
        print(f"Best Fitness Score (Run {i+1}): {best_score_run:.2f}")
        plot_pso_convergence(history_run, title=f"Optimization Run {i+1}")

        fuzzy_system_factory = ParameterizedFuzzySystemFactory()
        optimized_fuzzy_controller_run = FuzzyTrafficController(fuzzy_system_factory, best_params_run)
        num_simulations_optimization_check = 50
        traffic_data_list_optimization_check = [{"A_density": random.randint(0, 100), "B_density": random.randint(0, 100)} for _ in range(num_simulations_optimization_check)]
        optimized_results_run = run_simulations(simulator, optimized_fuzzy_controller_run, traffic_data_list_optimization_check)
        all_optimized_results.append(optimized_results_run.mean())
        print(f"Optimized Fuzzy (Run {i+1}) - Average Metrics:\n{optimized_results_run.mean()}")

    # --- Run Simulations for the Optimized Controller (using best_params from the initial run) ---
    initial_best_params, _, _ = optimize_fuzzy_system(fitness_evaluator, lb, ub)
    fuzzy_system_factory = ParameterizedFuzzySystemFactory()
    optimized_fuzzy_controller = FuzzyTrafficController(fuzzy_system_factory, initial_best_params)
    fixed_time_controller = FixedTimeTrafficController(30, 30)
    fuzzy_results = run_simulations(simulator, optimized_fuzzy_controller, traffic_data_list)
    fixed_time_results = run_simulations(simulator, fixed_time_controller, traffic_data_list)

    # --- Statistical Analysis of Simulation Results ---
    print("\n--- Statistical Analysis of Simulation Results ---\n")
    print("Descriptive Statistics for Optimized Fuzzy Controller Performance:")
    print(fuzzy_results.describe())
    save_table_as_png(fuzzy_results.describe().reset_index(), 'fuzzy_descriptive_stats.png')
    print("\nDescriptive Statistics for Fixed Time Controller Performance:")
    print(fixed_time_results.describe())
    save_table_as_png(fixed_time_results.describe().reset_index(), 'fixed_time_descriptive_stats.png')

    # --- Hypothesis Testing: Paired T-test Comparison of Controller Performance ---
    print("\n--- Hypothesis Testing: Paired T-test Comparison of Controller Performance ---")
    alpha = 0.05
    ttest_results = []
    for col in fuzzy_results.columns:
        if col not in ['total_vehicles_A_served', 'total_vehicles_B_served']:
            t_stat, p_value = stats.ttest_rel(fuzzy_results[col], fixed_time_results[col])
            mean_diff = fuzzy_results[col].mean() - fixed_time_results[col].mean()
            conclusion = ""
            if p_value < alpha:
                conclusion = f"Optimized Fuzzy has a significantly {'higher' if mean_diff > 0 else 'lower'} mean for this metric."
            else:
                conclusion = "No statistically significant difference observed between the controllers for this metric."
            ttest_results.append({"Metric": col.replace("_", " ").title(), "T-statistic": f"{t_stat:.4f}", "P-value": f"{p_value:.4f}", "Significance": "Yes" if p_value < alpha else "No", "Conclusion": conclusion})
    ttest_df = pd.DataFrame(ttest_results)
    print(ttest_df)
    save_table_as_png(ttest_df, 'ttest_results.png')

    # --- Visualization of Simulation Results and Comparisons ---
    print("\n--- Visualization of Simulation Results and Comparisons ---\n")
    sns.set_style("whitegrid")
    os.makedirs('graphs', exist_ok=True)
    graph_dir = 'graphs'

    for col in fuzzy_results.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(fuzzy_results[col], kde=True, label='Optimized Fuzzy Controller', color='skyblue')
        sns.histplot(fixed_time_results[col], kde=True, label='Fixed Time Controller', color='salmon')
        plt.title(f'Distribution of {col.replace("_", " ").title()} Across Simulations')
        plt.xlabel(col.replace("_", " ").title())
        plt.ylabel('Frequency')
        plt.legend()
        plt.savefig(os.path.join(graph_dir, f'histogram_{col}.png'))
        plt.close()

    metrics_to_plot = ['total_wait_time', 'average_throughput', 'wait_time_disparity', 'average_system_time']
    data_to_plot = pd.melt(fuzzy_results[metrics_to_plot], var_name='Metric', value_name='Value')
    data_to_plot['Controller'] = 'Optimized Fuzzy Controller'
    data_fixed = pd.melt(fixed_time_results[metrics_to_plot], var_name='Metric', value_name='Value')
    data_fixed['Controller'] = 'Fixed Time Controller'
    combined_data = pd.concat([data_to_plot, data_fixed])

    plt.figure(figsize=(12, 8))
    sns.boxplot(x='Metric', y='Value', hue='Controller', data=combined_data)
    plt.title('Comparison of Performance Metrics Between Controllers')
    plt.xlabel('Performance Metric')
    plt.ylabel('Value')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(graph_dir, 'boxplot_comparison.png'))
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='average_system_time', y='average_throughput', data=fuzzy_results, label='Optimized Fuzzy Controller', color='skyblue')
    sns.scatterplot(x='average_system_time', y='average_throughput', data=fixed_time_results, label='Fixed Time Controller', color='salmon')
    plt.title('Relationship between Average System Time and Average Throughput')
    plt.xlabel('Average System Time')
    plt.ylabel('Average Throughput')
    plt.legend()
    plt.savefig(os.path.join(graph_dir, 'scatterplot_system_time_throughput.png'))
    plt.close()

    # --- Comparative Analysis Summary of Average Performance Metrics ---
    print("\n--- Comparative Analysis Summary of Average Performance Metrics ---\n")
    optimized_avg = fuzzy_results.mean()
    fixed_avg = fixed_time_results.mean()
    comparison_data = []
    for col in fuzzy_results.columns:
        if col not in ['total_vehicles_A_served', 'total_vehicles_B_served']:
            comparison_data.append({"Metric": col.replace("_", " ").title(), "Optimized Fuzzy Controller (Mean)": f"{optimized_avg[col]:.2f}", "Fixed Time Controller (Mean)": f"{fixed_avg[col]:.2f}"})
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df)
    save_table_as_png(comparison_df, 'comparison_summary.png')

    print("\n--- Paired T-test Results Summary ---\n")
    print(ttest_df)

    print("\nConsider the Paired T-test Results Summary to conclude whether the Optimized Fuzzy Controller performed significantly better or worse than the Fixed Time Controller for each metric, given the same traffic scenarios.")