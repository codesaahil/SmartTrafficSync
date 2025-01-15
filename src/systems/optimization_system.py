# src/systems/optimization_system.py

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from pyswarm import pso

class FuzzyTrafficController:
    def __init__(self, car_system):
        self.car_system = car_system
        self.best_params = None
        self.fuzzy_system = None
        self.optimize_fuzzy_system()

    def create_fuzzy_system(self, params):
        # Ensure parameters are sorted to satisfy a <= b <= c
        low_a, med_a, high_a = sorted(params[:3])
        low_b, med_b, high_b = sorted(params[3:6])

        # Inputs
        A_density = ctrl.Antecedent(np.arange(0, 6, 1), 'A_density')
        B_density = ctrl.Antecedent(np.arange(0, 6, 1), 'B_density')

        # Outputs
        A_duration = ctrl.Consequent(np.arange(5, 21, 1), 'A_duration')
        B_duration = ctrl.Consequent(np.arange(5, 21, 1), 'B_duration')

        # Membership functions for densities
        A_density['low'] = fuzz.trapmf(A_density.universe, [0, 0, low_a, med_a])
        A_density['medium'] = fuzz.trimf(A_density.universe, [low_a, med_a, high_a])
        A_density['high'] = fuzz.trapmf(A_density.universe, [med_a, high_a, 100, 100])

        B_density['low'] = fuzz.trapmf(B_density.universe, [0, 0, low_b, med_b])
        B_density['medium'] = fuzz.trimf(B_density.universe, [low_b, med_b, high_b])
        B_density['high'] = fuzz.trapmf(B_density.universe, [med_b, high_b, 100, 100])

        # Membership functions for durations
        A_duration['short'] = fuzz.trimf(A_duration.universe, [5, 8, 12])
        A_duration['medium'] = fuzz.trimf(A_duration.universe, [8, 15, 18])
        A_duration['long'] = fuzz.trimf(A_duration.universe, [15, 18, 20])

        B_duration['short'] = fuzz.trimf(B_duration.universe, [5, 8, 12])
        B_duration['medium'] = fuzz.trimf(B_duration.universe, [8, 15, 18])
        B_duration['long'] = fuzz.trimf(B_duration.universe, [15, 18, 20])

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

    def simulate_traffic(self, params):
        fuzzy_system = self.create_fuzzy_system(params)

        a_density_value = min(100, max(0, self.car_system.get_traffic_density_horizontal()))
        b_density_value = min(100, max(0, self.car_system.get_traffic_density_vertical()))

        fuzzy_system.input['A_density'] = a_density_value
        fuzzy_system.input['B_density'] = b_density_value

        try:
            fuzzy_system.compute()
            a_duration = fuzzy_system.output['A_duration']
            b_duration = fuzzy_system.output['B_duration']
            return a_duration, b_duration
        except Exception as e:
            print(f"Error in fuzzy computation: {e}")
            return None, None

    def fitness_function(self, params):
        a_duration, b_duration = self.simulate_traffic(params)

        if a_duration is None or b_duration is None:
            return float('inf')  # Penalize invalid outputs

        a_density_value = np.clip(self.car_system.get_traffic_density_horizontal(), 0, 100)
        b_density_value = np.clip(self.car_system.get_traffic_density_vertical(), 0, 100)
        epsilon = 1e-9

        total_wait_time_a = a_density_value * b_duration
        total_wait_time_b = b_density_value * a_duration
        total_wait_time = total_wait_time_a + total_wait_time_b

        vehicles_a_served = min(a_density_value, (a_duration / 60) * 100)
        vehicles_b_served = min(b_density_value, (b_duration / 60) * 100)
        total_vehicles_served = vehicles_a_served + vehicles_b_served

        average_system_time = total_wait_time / (total_vehicles_served + epsilon) if total_vehicles_served > 0 else float('inf')
        average_throughput = total_vehicles_served / (a_density_value + b_density_value + epsilon) if (a_density_value + b_density_value) > 0 else 0

        weight_wait = 1.0
        weight_throughput = 1.0

        fitness = weight_wait * average_system_time - weight_throughput * average_throughput
        return fitness

    def optimize_fuzzy_system(self):
        # Parameter bounds for fuzzy membership functions
        lb = [0, 2, 4, 0, 2, 4]  # Lower bounds for low, medium, high (A and B)
        ub = [2, 3, 6, 2, 3, 6]  # Upper bounds for low, medium, high (A and B)

        # Run PSO
        self.best_params, _ = pso(
            self.fitness_function, 
            lb, 
            ub, 
            swarmsize=20, 
            maxiter=50
        )

        # Create fuzzy system with optimized parameters
        self.fuzzy_system = self.create_fuzzy_system(self.best_params)

    def get_traffic_light_durations(self):
        """
        Get optimized traffic light durations based on current traffic density
        
        Returns:
        tuple: (horizontal_duration, vertical_duration)
        """
        if not self.fuzzy_system:
            return (30, 30)  # Default if not optimized

        # Get current traffic densities
        A_density_value = min(100, max(0, self.car_system.get_traffic_density_horizontal()))
        B_density_value = min(100, max(0, self.car_system.get_traffic_density_vertical()))

        # Set inputs to fuzzy system
        self.fuzzy_system.input['A_density'] = A_density_value
        self.fuzzy_system.input['B_density'] = B_density_value

        try:
            # Compute fuzzy outputs
            self.fuzzy_system.compute()

            # Get and constrain durations
            A_duration = max(10, min(60, self.fuzzy_system.output['A_duration']))
            B_duration = max(10, min(60, self.fuzzy_system.output['B_duration']))

            return (A_duration, B_duration)
        except Exception as e:
            print(f"Error getting traffic light durations: {e}")
            return (30, 30)  # Default durations
        
    def get_all_states(self, durations):
        A_green_duration = durations[0] * 60
        B_green_duration = durations[1] * 60
        A_yellow_duration = 3 * 60
        B_yellow_duration = 3 * 60
        A_red_duration = B_green_duration - A_yellow_duration
        B_red_duration = A_green_duration - B_yellow_duration

        A_light = {
            "green": A_green_duration,
            "yellow": A_yellow_duration,
            "red": A_red_duration
        }

        B_light = {
            "green": B_green_duration,
            "yellow": B_yellow_duration,
            "red": B_red_duration
        }

        return [
            A_light,
            B_light,
            B_light,
            A_light,
            A_light,
            B_light,
            B_light,
            A_light
        ]

    def reoptimize_periodically(self, iteration_count):
        """
        Periodically reoptimize the fuzzy system based on collected traffic data
        
        Args:
            iteration_count (int): Current game iteration
        """
        # Reoptimize every 1000 iterations, for example
        if iteration_count % 1000 == 0:
            self.optimize_fuzzy_system()