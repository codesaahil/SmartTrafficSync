import numpy as np
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class TrafficSystem:
    """
    Data class representing the traffic system parameters.

    Attributes:
        num_intersections (int): Number of intersections in the traffic system.
        intersection_lengths (np.ndarray): Lengths of each intersection in meters.
        distances (np.ndarray): Distances between consecutive intersections in meters.
        avg_speed (float): Average vehicle speed in meters per second.
        min_signal_time (int): Minimum green/red signal time in seconds.
        max_signal_time (int): Maximum green/red signal time in seconds.
        yellow_time (int): Fixed yellow signal time in seconds (default: 3 seconds).
        traffic_volume (float): Average traffic volume (vehicles per second).
    """

    num_intersections: int
    intersection_lengths: np.ndarray  # Length of each intersection in meters
    distances: np.ndarray  # Distances between consecutive intersections in meters
    avg_speed: float  # Average vehicle speed in m/s
    min_signal_time: int  # Minimum green/red signal time in seconds
    max_signal_time: int  # Maximum green/red signal time in seconds
    yellow_time: int = 3  # Fixed yellow signal time in seconds
    traffic_volume: float = 0.3  # vehicles per second (default: moderate traffic)


def update_position(wolf_pos: float, leader_pos: float, a: float) -> float:
    """
    Update the position of a wolf based on the leader's position.

    Parameters:
        wolf_pos (float): Current position of the wolf.
        leader_pos (float): Position of the leader wolf.
        a (float): Coefficient controlling the balance between exploration and exploitation.

    Returns:
        float: Updated position of the wolf, rounded to the nearest integer.
    """
    rng = np.random.default_rng(1)
    r1, r2 = rng.random(), rng.random()
    a = 2 * a * r1 - a
    c = 2 * r2
    d = abs(c * leader_pos - wolf_pos)
    return round(leader_pos - a * d)


class TrafficLightOptimizer:
    """
    Class for optimizing traffic light timings using the Grey Wolf Optimization algorithm.

    Attributes:
        system (TrafficSystem): The traffic system parameters.
        search_space (np.ndarray): The search space defining the bounds for signal timings and offsets.
    """

    def __init__(self, traffic_system: TrafficSystem):
        self.system = traffic_system
        # Search space for each intersection: [green_time, offset]
        self.search_space = np.array(
            [
                [
                    self.system.min_signal_time,
                    self.system.max_signal_time,
                ],  # bounds for green time
                [0, self.system.max_signal_time * 2],  # bounds for offset
            ]
        )
        self.search_space = np.tile(
            self.search_space, (self.system.num_intersections, 1)
        )

    def initialize_wolves(self, num_wolves: int) -> np.ndarray:
        """
        Initialize wolves with random signal timings and offsets.

        Parameters:
            num_wolves (int): Number of wolves to initialize.

        Returns:
            np.ndarray: Randomly initialized wolf positions (signal timings and offsets).
        """
        rng = np.random.default_rng(1)
        dimensions = self.search_space.shape[0]
        wolves = rng.integers(
            self.search_space[:, 0],
            self.search_space[:, 1] + 1,
            size=(num_wolves, dimensions),
        )
        return wolves

    def calculate_vehicle_delays(
        self, signal_timings: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        Calculate delays, throughput, and queue lengths for vehicles.

        Parameters:
            signal_timings (np.ndarray): The current signal timings and offsets.

        Returns:
            tuple: (average_delay, throughput, max_queue_length)
        """
        total_delay = 0
        vehicles_processed = 0
        max_queue_length = 0
        simulation_time = 3600  # Simulate for 1 hour

        # Vehicle parameters
        vehicle_length = 5  # meters
        safe_distance = 2  # meters
        vehicle_spacing = vehicle_length + safe_distance

        for i in range(self.system.num_intersections):
            green_time = signal_timings[i * 2]
            offset = signal_timings[i * 2 + 1]
            red_time = green_time  # Equal red and green times
            cycle_time = green_time + self.system.yellow_time + red_time

            # Maximum flow rate during green time (vehicles/second)
            max_flow_rate = self.system.avg_speed / vehicle_spacing

            # Calculate queue formation during red and yellow time
            queue_time = red_time + self.system.yellow_time
            max_queue = self.system.traffic_volume * queue_time
            max_queue_length = max(max_queue_length, int(max_queue * vehicle_spacing))

            # Calculate effective green time (considering startup loss and clearance)
            startup_loss = 2  # seconds
            effective_green = max(0, int(green_time - startup_loss))

            # Calculate vehicles that can pass during effective green time
            vehicles_per_green = min(
                max_flow_rate * effective_green,  # Flow capacity
                float(
                    max_queue + (self.system.traffic_volume * effective_green)
                ),  # Available vehicles
            )

            # Calculate coordination efficiency with previous intersection
            if i > 0:
                prev_offset = signal_timings[(i - 1) * 2 + 1]
                travel_time = self.system.distances[i - 1] / self.system.avg_speed
                ideal_offset = travel_time % cycle_time
                offset_difference = abs(offset - prev_offset - ideal_offset)
                coordination_factor = 1 - (
                    min(int(offset_difference), int(cycle_time / 2)) / (cycle_time / 2)
                )
                vehicles_per_green *= 0.7 + (0.3 * coordination_factor)

            # Calculate delays
            average_queue = max_queue / 2  # Average queue during red time
            total_delay += (average_queue * queue_time) + (
                vehicles_per_green * startup_loss / 2
            )

            # Calculate throughput
            num_cycles = simulation_time / cycle_time
            vehicles_processed += vehicles_per_green * num_cycles

        average_delay = (
            total_delay / vehicles_processed if vehicles_processed > 0 else float("inf")
        )
        throughput = vehicles_processed / simulation_time

        return average_delay, throughput, max_queue_length

    def fitness_function(self, signal_timings: np.ndarray) -> float:
        """
        Evaluate the fitness of the current signal timings.

        Parameters:
            signal_timings (np.ndarray): The current signal timings and offsets.

        Returns:
            float: A fitness score indicating the performance of the signal timings.
        """
        avg_delay, throughput, max_queue = self.calculate_vehicle_delays(signal_timings)

        # Calculate cycle efficiency (penalize both very short and very long cycles)
        cycle_times = []
        for i in range(self.system.num_intersections):
            green_time = signal_timings[i * 2]
            cycle_time = 2 * green_time + self.system.yellow_time
            optimal_cycle = (
                self.system.max_signal_time + self.system.min_signal_time
            ) / 2
            cycle_efficiency = 1 - abs(cycle_time - optimal_cycle) / optimal_cycle
            cycle_times.append(cycle_efficiency)

        avg_cycle_efficiency = np.mean(cycle_times)

        # Calculate green wave coordination score
        coordination_score = 0
        if self.system.num_intersections > 1:
            for i in range(1, self.system.num_intersections):
                prev_offset = signal_timings[(i - 1) * 2 + 1]
                curr_offset = signal_timings[i * 2 + 1]
                travel_time = self.system.distances[i - 1] / self.system.avg_speed
                ideal_offset = travel_time % (
                    signal_timings[i * 2] * 2 + self.system.yellow_time
                )
                offset_diff = abs(curr_offset - prev_offset - ideal_offset)
                coordination_score += 1 - min(
                    int(offset_diff / 30), 1
                )  # Normalize to [0,1]
            coordination_score /= self.system.num_intersections - 1
        else:
            coordination_score = 1

        # Normalize metrics
        norm_delay = min(avg_delay / 120.0, 1.0)  # Normalize to max 2 minutes delay
        norm_throughput = min(throughput / (self.system.traffic_volume * 1.5), 1.0)
        norm_queue = min(max_queue / 100, 1.0)  # Normalize to 100m max queue

        # Combined fitness (lower is better)
        weights = {
            "delay": 0.35,
            "throughput": 0.25,
            "queue": 0.15,
            "coordination": 0.15,
            "cycle_efficiency": 0.10,
        }

        fitness = (
            weights["delay"] * norm_delay
            + weights["throughput"] * (1 - norm_throughput)
            + weights["queue"] * norm_queue
            + weights["coordination"] * (1 - coordination_score)
            + weights["cycle_efficiency"] * (1 - avg_cycle_efficiency)
        )

        return fitness

    def update_wolves(
        self,
        wolves: np.ndarray,
        alpha: np.ndarray,
        beta: np.ndarray,
        gamma: np.ndarray,
        a: float,
    ) -> None:
        """
        Update positions of all wolves in the search space.

        Parameters:
            wolves (np.ndarray): The current positions of the wolves.
            alpha (np.ndarray): Position of the best wolf (alpha).
            beta (np.ndarray): Position of the second-best wolf (beta).
            gamma (np.ndarray): Position of the third-best wolf (gamma).
            a (float): Coefficient controlling the balance between exploration and exploitation.
        """
        for i, wolf in enumerate(wolves):
            for j in range(len(self.search_space)):
                x1 = update_position(wolf[j], float(alpha[j]), a)
                x2 = update_position(wolf[j], float(beta[j]), a)
                x3 = update_position(wolf[j], float(gamma[j]), a)

                wolves[i, j] = int(round((x1 + x2 + x3) / 3))
                wolves[i, j] = np.clip(
                    wolves[i, j], self.search_space[j, 0], self.search_space[j, 1]
                )

    def optimize(self, num_wolves: int = 30, max_iterations: int = 100) -> np.ndarray:
        """
        Run the Grey Wolf Optimization algorithm to find optimal signal timings.

        Parameters:
            num_wolves (int): Number of wolves to use in the optimization (default: 30).
            max_iterations (int): Maximum number of iterations for the optimization (default: 100).

        Returns:
            np.ndarray: Best signal timings and offsets found during optimization.
        """
        wolves = self.initialize_wolves(num_wolves)
        dimensions = len(self.search_space)
        alpha = np.zeros(dimensions, dtype=int)
        beta = np.zeros(dimensions, dtype=int)
        gamma = np.zeros(dimensions, dtype=int)

        alpha_score = float("inf")
        beta_score = float("inf")
        gamma_score = float("inf")

        for iteration in range(max_iterations):
            a = 2 - (2 * iteration / max_iterations)

            for wolf in wolves:
                fitness = self.fitness_function(wolf)
                if fitness < alpha_score:
                    alpha_score = fitness
                    alpha = wolf.copy()
                elif fitness < beta_score:
                    beta_score = fitness
                    beta = wolf.copy()
                elif fitness < gamma_score:
                    gamma_score = fitness
                    gamma = wolf.copy()

            self.update_wolves(wolves, alpha, beta, gamma, a)

        return alpha


def format_solution(
    solution: np.ndarray, num_intersections: int, yellow_time: int
) -> List[dict]:
    """
    Format the solution into a readable format for traffic lights.

    Parameters:
        solution (np.ndarray): The optimal signal timings and offsets.
        num_intersections (int): Number of intersections.
        yellow_time (int): Duration of the yellow signal in seconds.

    Returns:
        List[dict]: List of dictionaries containing formatted traffic light timings.
    """
    traffic_lights = []
    for i in range(num_intersections):
        green_time = solution[i * 2]
        offset = solution[i * 2 + 1]

        traffic_lights.append(
            {
                "intersection": i + 1,
                "green_time": green_time,
                "yellow_time": yellow_time,
                "red_time": green_time,
                "offset": offset,
                "cycle_time": 2 * green_time + yellow_time,
            }
        )

    return traffic_lights


if __name__ == "__main__":
    # Example traffic system parameters
    system = TrafficSystem(
        num_intersections=5,
        intersection_lengths=np.array([30, 25, 35, 10, 50]),  # meters
        distances=np.array([200, 250, 100, 300, 200]),  # meters between intersections
        avg_speed=13.89,  # 50 km/h in m/s
        min_signal_time=30,  # seconds
        max_signal_time=90,  # seconds
        yellow_time=3,  # seconds
        traffic_volume=0.3,  # vehicles per second (moderate traffic)
    )

    # Create optimizer and run optimization
    system_optimizer = TrafficLightOptimizer(system)
    best_solution = system_optimizer.optimize(num_wolves=30, max_iterations=100)

    # Format and print results
    optimal_traffic_lights = format_solution(
        best_solution, system.num_intersections, system.yellow_time
    )

    print("\nOptimized Traffic Light Timings:")
    for light in optimal_traffic_lights:
        print(f"\nIntersection {light['intersection']}:")
        print(f"  Green Time: {light['green_time']} seconds")
        print(f"  Yellow Time: {light['yellow_time']} seconds")
        print(f"  Red Time: {light['red_time']} seconds")
        print(f"  Offset: {light['offset']} seconds")
        print(f"  Total Cycle: {light['cycle_time']} seconds")

    # Calculate and print performance metrics
    optimal_avg_delay, optimal_throughput, optimal_max_queue = (
        system_optimizer.calculate_vehicle_delays(best_solution)
    )
    print("\nPerformance Metrics:")
    print(f"Average Delay: {optimal_avg_delay:.2f} seconds per vehicle")
    print(f"Throughput: {optimal_throughput * 3600:.0f} vehicles per hour")
    print(f"Maximum Queue Length: {optimal_max_queue:.1f} meters")
