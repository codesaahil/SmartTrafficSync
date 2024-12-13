import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from pyswarm import pso
import random

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
        # Use actual car system data instead of random generation
        fuzzy_system = self.create_fuzzy_system(params)

        # Get traffic densities from car system
        A_density_value = min(100, max(0, self.car_system.get_traffic_density_horizontal()))
        B_density_value = min(100, max(0, self.car_system.get_traffic_density_vertical()))

        # Compute fuzzy system output
        fuzzy_system.input['A_density'] = A_density_value
        fuzzy_system.input['B_density'] = B_density_value

        try:
            fuzzy_system.compute()
            A_duration = fuzzy_system.output['A_duration']
            B_duration = fuzzy_system.output['B_duration']

            # Simple fitness calculation
            # Considers density, duration, and potential traffic flow
            fitness = (A_density_value * B_duration + B_density_value * A_duration) / (A_duration + B_duration)
            return fitness
        except Exception as e:
            print(f"Error in fuzzy computation: {e}")
            return 0

    def fitness_function(self, params):
        # Negative fitness for PSO minimization
        return -self.simulate_traffic(params)

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