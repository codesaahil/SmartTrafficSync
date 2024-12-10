import random
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from deap import base, creator, tools, algorithms
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class TrafficSystem:
    """
    A class to represent a traffic system with various parameters for optimizing
    traffic light signals at multiple intersections.

    Attributes:
        num_intersections (int): The total number of intersections in the traffic system.
        intersection_lengths (np.ndarray): An array representing the lengths of each intersection in meters.
        distances (np.ndarray): An array representing the distances between intersections in meters.
        avg_speed (float): The average speed of vehicles in meters per second.
        min_signal_time (int): The minimum duration for traffic signals to be green in seconds.
        max_signal_time (int): The maximum duration for traffic signals to be green in seconds.
        yellow_time (int): The duration for yellow signals in seconds (default is 3 seconds).
        traffic_volume (float): The average number of vehicles passing through an intersection per second (default is 0.3).

    Methods:
        __post_init__(): Performs checks and initialization after the default `__init__` method.
    """

    num_intersections: int
    intersection_lengths: np.ndarray  # meters
    distances: np.ndarray  # meters between intersections
    avg_speed: float  # m/s
    min_signal_time: int  # seconds
    max_signal_time: int  # seconds
    yellow_time: int = 3  # seconds
    traffic_volume: float = 0.3  # vehicles per second


class FuzzyTrafficController:
    """
    A class to implement a fuzzy logic traffic controller for managing traffic signals
    based on queue length, arrival rate, and waiting time at intersections.

    Attributes:
        queue_length (ctrl.Antecedent): Fuzzy input variable representing the length of the queue in vehicles.
        arrival_rate (ctrl.Antecedent): Fuzzy input variable representing the rate of vehicle arrivals (vehicles per second).
        waiting_time (ctrl.Antecedent): Fuzzy input variable representing the time vehicles have been waiting at the intersection.
        green_time (ctrl.Consequent): Fuzzy output variable representing the duration of the green signal (seconds).
        offset (ctrl.Consequent): Fuzzy output variable representing the offset time for signal synchronization (seconds).
        rules (list): A list of fuzzy rules for determining green time and offset based on input variables.
        control_system (ctrl.ControlSystem): The fuzzy control system containing the defined rules.
        controller (ctrl.ControlSystemSimulation): An instance for simulating the control system.

    Methods:
        __init__(): Initializes the fuzzy variables, membership functions, and rules for the traffic controller.
    """

    def __init__(self):
        """Initializes the fuzzy variables and rules for the traffic controller."""

        # Input variables
        self.queue_length = ctrl.Antecedent(np.arange(0, 101, 1), "queue_length")
        self.arrival_rate = ctrl.Antecedent(np.arange(0, 1.1, 0.1), "arrival_rate")
        self.waiting_time = ctrl.Antecedent(np.arange(0, 121, 1), "waiting_time")

        # Output variables
        self.green_time = ctrl.Consequent(np.arange(30, 91, 1), "green_time")
        self.offset = ctrl.Consequent(np.arange(0, 91, 1), "offset")

        # Membership functions for queue length
        self.queue_length["short"] = fuzz.trimf(self.queue_length.universe, [0, 0, 30])
        self.queue_length["medium"] = fuzz.trimf(
            self.queue_length.universe, [20, 50, 80]
        )
        self.queue_length["long"] = fuzz.trimf(
            self.queue_length.universe, [70, 100, 100]
        )

        # Membership functions for arrival rate
        self.arrival_rate["low"] = fuzz.trimf(self.arrival_rate.universe, [0, 0, 0.4])
        self.arrival_rate["medium"] = fuzz.trimf(
            self.arrival_rate.universe, [0.3, 0.5, 0.7]
        )
        self.arrival_rate["high"] = fuzz.trimf(self.arrival_rate.universe, [0.6, 1, 1])

        # Membership functions for waiting time
        self.waiting_time["short"] = fuzz.trimf(self.waiting_time.universe, [0, 0, 40])
        self.waiting_time["medium"] = fuzz.trimf(
            self.waiting_time.universe, [30, 60, 90]
        )
        self.waiting_time["long"] = fuzz.trimf(
            self.waiting_time.universe, [80, 120, 120]
        )

        # Membership functions for green time
        self.green_time["short"] = fuzz.trimf(self.green_time.universe, [30, 30, 50])
        self.green_time["medium"] = fuzz.trimf(self.green_time.universe, [40, 60, 80])
        self.green_time["long"] = fuzz.trimf(self.green_time.universe, [70, 90, 90])

        # Membership functions for offset
        self.offset["small"] = fuzz.trimf(self.offset.universe, [0, 0, 30])
        self.offset["medium"] = fuzz.trimf(self.offset.universe, [20, 45, 70])
        self.offset["large"] = fuzz.trimf(self.offset.universe, [60, 90, 90])

        # Define fuzzy rules
        self.rules = [
            # Rules for green time
            ctrl.Rule(
                self.queue_length["long"] & self.arrival_rate["high"],
                self.green_time["long"],
            ),
            ctrl.Rule(
                self.queue_length["medium"] & self.arrival_rate["medium"],
                self.green_time["medium"],
            ),
            ctrl.Rule(
                self.queue_length["short"] & self.arrival_rate["low"],
                self.green_time["short"],
            ),
            ctrl.Rule(self.waiting_time["long"], self.green_time["long"]),
            # Rules for offset
            ctrl.Rule(
                self.arrival_rate["high"] & self.queue_length["long"],
                self.offset["medium"],
            ),
            ctrl.Rule(
                self.arrival_rate["medium"] & self.queue_length["medium"],
                self.offset["medium"],
            ),
            ctrl.Rule(
                self.arrival_rate["low"] & self.queue_length["short"],
                self.offset["small"],
            ),
        ]

        # Create control system
        self.control_system = ctrl.ControlSystem(self.rules)
        self.controller = ctrl.ControlSystemSimulation(self.control_system)

    def compute(
        self, queue_length: float, arrival_rate: float, waiting_time: float
    ) -> Tuple[float, float]:
        """
        Compute the green time and offset for traffic signals using fuzzy logic based on
        the current queue length, arrival rate of vehicles, and waiting time at the intersection.

        Parameters:
            queue_length (float): The current length of the queue at the intersection (in vehicles).
            arrival_rate (float): The rate of vehicle arrivals at the intersection (in vehicles per second).
            waiting_time (float): The time that vehicles have been waiting at the intersection (in seconds).

        Returns:
            Tuple[float, float]: A tuple containing the computed green time (seconds) and offset (seconds)
                                 for the traffic signals.

        Raises:
            Exception: If there is an error during the fuzzy computation, an exception will be raised.
        """
        self.controller.input["queue_length"] = queue_length
        self.controller.input["arrival_rate"] = arrival_rate
        self.controller.input["waiting_time"] = waiting_time

        try:
            self.controller.compute()
            green_time = self.controller.output["green_time"]
            offset = self.controller.output["offset"]
        except Exception as e:
            print("Fuzzy computation error")
            raise e

        return green_time, offset


class FuzzyTrafficOptimizer:
    """
    A class to optimize traffic signal timing using fuzzy logic and genetic algorithms.

    This class integrates a fuzzy traffic controller with a genetic algorithm to
    optimize parameters such as queue length thresholds for traffic signals based
    on a given traffic system model.

    Attributes:
        system (TrafficSystem): An instance of the TrafficSystem class containing
                                traffic parameters for optimization.
        fuzzy_controller (FuzzyTrafficController): An instance of the FuzzyTrafficController
                                                   to manage fuzzy logic operations.
        toolbox (deap.base.Toolbox): A DEAP toolbox to facilitate genetic algorithm operations.

    Methods:
        __init__(traffic_system: TrafficSystem): Initializes the FuzzyTrafficOptimizer with
                                                  a traffic system and sets up DEAP components.
    """

    def __init__(self, traffic_system: TrafficSystem):
        """
        Initializes the FuzzyTrafficOptimizer with a traffic system and sets up
        the DEAP framework for optimization.

        Parameters:
            traffic_system (TrafficSystem): An instance of the TrafficSystem class,
                                             providing the context for traffic optimization.

        Initializes the fuzzy controller and configures the DEAP framework,
        including creating fitness types, individual structures, and genetic
        algorithm operators for optimization.
        """
        self.system = traffic_system
        self.fuzzy_controller = FuzzyTrafficController()

        # Create DEAP types
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        # Initialize toolbox
        self.toolbox = base.Toolbox()

        # Attribute generator for queue length threshold
        self.toolbox.register("attr_float", random.uniform, 0, 100)

        # Structure initializers
        self.toolbox.register(
            "individual",
            tools.initRepeat,
            creator.Individual,
            self.toolbox.attr_float,
            n=3,
        )  # 3 parameters to optimize
        self.toolbox.register(
            "population", tools.initRepeat, list, self.toolbox.individual
        )

        # Operator registering
        self.toolbox.register("evaluate", self.evaluate)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=10, indpb=0.2)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def calculate_traffic_metrics(
        self, signal_timings: List[dict]
    ) -> Tuple[float, float, float]:
        """
        Calculate traffic metrics based on the provided signal timings for each intersection.

        This method computes average delay, hourly throughput, and maximum queue length
        for a traffic system based on the signal timings of each intersection.

        Parameters:
            signal_timings (List[dict]): A list of dictionaries, each containing the
                                          'cycle_time' and 'green_time' for an intersection.
                                          Example: [{'cycle_time': 60, 'green_time': 30}, ...]

        Returns:
            Tuple[float, float, float]: A tuple containing:
                - avg_delay (float): The average delay per vehicle (in seconds).
                - hourly_throughput (float): The total number of vehicles that can pass
                  through the intersection in one hour.
                - max_queue (float): The maximum queue length (in vehicles) observed
                  during the simulation.

        Raises:
            ValueError: If the length of signal_timings does not match the number of intersections
                        in the traffic system.
        """
        total_delay = 0
        total_throughput = 0
        max_queue = 0
        simulation_time = 3600  # 1 hour simulation

        if len(signal_timings) != self.system.num_intersections:
            raise ValueError(
                "The length of signal_timings must match the number of intersections."
            )

        for i in range(self.system.num_intersections):
            cycle_time = signal_timings[i]["cycle_time"]
            green_time = signal_timings[i]["green_time"]

            # Calculate queue formation
            arrival_rate = self.system.traffic_volume
            max_queue_length = arrival_rate * (cycle_time - green_time)
            max_queue = max(max_queue, max_queue_length)

            # Calculate delay
            avg_delay = (cycle_time - green_time) / 2
            total_delay += avg_delay * arrival_rate * simulation_time

            # Calculate throughput
            max_flow_rate = 1 / 2.5  # vehicles per second (assuming 2.5s headway)
            cycle_throughput = min(
                max_flow_rate * green_time, arrival_rate * cycle_time
            )
            total_throughput += cycle_throughput * (simulation_time / cycle_time)

        avg_delay = (
            total_delay / total_throughput if total_throughput > 0 else float("inf")
        )
        hourly_throughput = total_throughput

        return avg_delay, hourly_throughput, max_queue

    def evaluate(self, individual: List[float]) -> Tuple[float,]:
        """
        Evaluate the fitness of an individual in the genetic algorithm.

        This method computes the fitness of an individual by generating signal timings
        for each intersection using fuzzy logic and calculating performance metrics such
        as average delay, throughput, and maximum queue length.

        Parameters:
            individual (List[float]): A list representing the individual's parameters,
                                       expected to contain:
                                       - queue_threshold (float): The threshold for queue length.
                                       - an unused placeholder (float).
                                       - waiting_threshold (float): The threshold for waiting time.

        Returns:
            Tuple[float]: A tuple containing a single float value representing the fitness
                          of the individual, where lower values are better.

        Notes:
            The fitness calculation is based on the following metrics:
                - Average delay (normalized to minutes).
                - Throughput ratio (compared to maximum possible throughput).
                - Maximum queue length (normalized).

        Raises:
            ValueError: If the length of the individual does not match the expected number of parameters.
        """
        if len(individual) != 3:
            raise ValueError("Individual must contain exactly three parameters.")

        queue_threshold, _, waiting_threshold = individual
        signal_timings = []

        # Generate signal timings for each intersection using fuzzy logic
        for i in range(self.system.num_intersections):
            # Calculate inputs for fuzzy controller
            queue_length = min(100, int(queue_threshold * self.system.traffic_volume))
            arrival_rate = min(1.0, self.system.traffic_volume)
            waiting_time = min(120, int(waiting_threshold))

            # Get fuzzy controller output
            green_time, offset = self.fuzzy_controller.compute(
                queue_length, arrival_rate, waiting_time
            )

            # Ensure values are within bounds
            green_time = np.clip(
                green_time, self.system.min_signal_time, self.system.max_signal_time
            )
            offset = np.clip(offset, 0, self.system.max_signal_time * 2)

            signal_timings.append(
                {
                    "intersection": i + 1,
                    "green_time": int(round(green_time)),
                    "yellow_time": self.system.yellow_time,
                    "red_time": int(round(green_time)),
                    "offset": int(round(offset)),
                    "cycle_time": int(round(2 * green_time + self.system.yellow_time)),
                }
            )

        # Calculate performance metrics
        avg_delay, throughput, max_queue = self.calculate_traffic_metrics(
            signal_timings
        )

        # Fitness is a combination of metrics (lower is better)
        fitness = (
            avg_delay / 60.0  # normalize to minutes
            + (1 - throughput / (self.system.traffic_volume * 3600))  # throughput ratio
            + max_queue / 100.0
        )  # normalize queue length

        return (fitness,)

    def optimize(self, population_size: int = 50, generations: int = 50) -> List[dict]:
        """
        Run the optimization process to determine optimal traffic signal timings.

        This method employs a genetic algorithm to optimize the traffic signal timings
        for a specified number of generations and population size. It returns the
        calculated signal timings based on the best solution found during the
        optimization process.

        Parameters:
            population_size (int, optional): The size of the population in the genetic
                                              algorithm. Default is 50.
            generations (int, optional): The number of generations for the optimization
                                          process. Default is 50.

        Returns:
            List[dict]: A list of dictionaries containing the optimal signal timings
                         for each intersection, with each dictionary containing:
                         - 'intersection': The intersection number (1-indexed).
                         - 'green_time': The optimized green light duration (in seconds).
                         - 'yellow_time': The duration of the yellow light (in seconds).
                         - 'red_time': The optimized red light duration (in seconds).
                         - 'offset': The offset timing (in seconds).
                         - 'cycle_time': The total cycle time (in seconds).

        Notes:
            The genetic algorithm uses crossover and mutation strategies to explore
            the solution space, optimizing the traffic signal timings based on
            predefined fitness criteria.

        Raises:
            ValueError: If the population size or number of generations is less than 1.
        """
        if population_size < 1 or generations < 1:
            raise ValueError(
                "Population size and number of generations must be at least 1."
            )

        # Create initial population
        pop = self.toolbox.population(n=population_size)

        # Initialize statistics
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)

        # Run evolution
        final_pop, _ = algorithms.eaSimple(
            pop,
            self.toolbox,
            cxpb=0.7,  # crossover probability
            mutpb=0.2,  # mutation probability
            ngen=generations,
            stats=stats,
            verbose=True,
        )

        # Get best solution
        best_individual = tools.selBest(final_pop, k=1)[0]

        # Generate final signal timings using best parameters
        queue_threshold, _, waiting_threshold = best_individual
        signal_timings = []

        for i in range(self.system.num_intersections):
            queue_length = min(100, queue_threshold * self.system.traffic_volume)
            arrival_rate = min(1.0, self.system.traffic_volume)
            waiting_time = min(120, waiting_threshold)

            green_time, offset = self.fuzzy_controller.compute(
                queue_length, arrival_rate, waiting_time
            )

            green_time = np.clip(
                green_time, self.system.min_signal_time, self.system.max_signal_time
            )
            offset = np.clip(offset, 0, self.system.max_signal_time * 2)

            signal_timings.append(
                {
                    "intersection": i + 1,
                    "green_time": int(round(green_time)),
                    "yellow_time": self.system.yellow_time,
                    "red_time": int(round(green_time)),
                    "offset": int(round(offset)),
                    "cycle_time": int(round(2 * green_time + self.system.yellow_time)),
                }
            )

        return signal_timings


if __name__ == "__main__":
    """
    Main execution block for optimizing traffic light timings using fuzzy logic.

    This script initializes a traffic system with specified parameters, creates an instance
    of the FuzzyTrafficOptimizer, and runs the optimization process to determine optimal
    traffic light timings for multiple intersections.

    It also prints the optimized signal timings and calculates performance metrics,
    including average delay, throughput, and maximum queue length.

    Traffic System Parameters:
        - num_intersections (int): Number of intersections to optimize.
        - intersection_lengths (np.ndarray): Array containing lengths of each intersection in meters.
        - distances (np.ndarray): Array containing distances between intersections in meters.
        - avg_speed (float): Average speed of vehicles in meters per second (m/s).
        - min_signal_time (int): Minimum signal time for traffic lights in seconds.
        - max_signal_time (int): Maximum signal time for traffic lights in seconds.
        - yellow_time (int): Duration of the yellow signal in seconds.
        - traffic_volume (float): Average traffic volume in vehicles per second.

    Outputs:
        - Prints optimized traffic light timings for each intersection.
        - Prints calculated performance metrics including:
            - Average delay per vehicle (in seconds).
            - Throughput (in vehicles per hour).
            - Maximum queue length (in vehicles).
    """

    # Example traffic system parameters
    system = TrafficSystem(
        num_intersections=3,
        intersection_lengths=np.array([30, 25, 35]),  # meters
        distances=np.array([200, 250]),  # meters between intersections
        avg_speed=13.89,  # 50 km/h in m/s
        min_signal_time=30,  # seconds
        max_signal_time=90,  # seconds
        yellow_time=3,  # seconds
        traffic_volume=0.3,  # vehicles per second (moderate traffic)
    )

    # Create optimizer and run optimization
    system_optimizer = FuzzyTrafficOptimizer(system)
    optimal_signal_timings = system_optimizer.optimize(
        population_size=50, generations=50
    )

    # Print results
    print("\nOptimized Traffic Light Timings:")
    for timing in optimal_signal_timings:
        print(f"\nIntersection {timing['intersection']}:")
        print(f"  Green Time: {timing['green_time']} seconds")
        print(f"  Yellow Time: {timing['yellow_time']} seconds")
        print(f"  Red Time: {timing['red_time']} seconds")
        print(f"  Offset: {timing['offset']} seconds")
        print(f"  Total Cycle: {timing['cycle_time']} seconds")

    # Calculate and print performance metrics
    optimal_avg_delay, optimal_throughput, optimal_max_queue = (
        system_optimizer.calculate_traffic_metrics(optimal_signal_timings)
    )
    print("\nPerformance Metrics:")
    print(f"Average Delay: {optimal_avg_delay:.2f} seconds per vehicle")
    print(f"Throughput: {optimal_throughput:.0f} vehicles per hour")
    print(f"Maximum Queue Length: {optimal_max_queue:.1f} vehicles")
