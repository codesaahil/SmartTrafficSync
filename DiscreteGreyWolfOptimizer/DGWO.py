import numpy as np


def initialize_wolves(search_space: np.ndarray, num_wolves: int) -> np.ndarray:
    """
    Initialize the wolves' positions within the specified discrete search space.

    Parameters:
        search_space (np.ndarray): The search space boundaries for each dimension,
                                   where each row is [lower_bound, upper_bound].
        num_wolves (int): The number of wolves in the pack.

    Returns:
        np.ndarray: Initialized positions of wolves within the search space.
    """
    rng = np.random.default_rng(1)  # Initialize a new random number generator
    dimensions = search_space.shape[1]
    wolves = rng.integers(
        search_space[:, 0], search_space[:, 1] + 1, size=(num_wolves, dimensions)
    )
    return wolves


def fitness_function(
    position: np.ndarray, target: np.ndarray = np.array([5, 5])
) -> float:
    """
    Calculate the fitness of a given position by computing its Euclidean distance
    to the target point.

    Parameters:
        position (np.ndarray): The current position of a wolf.
        target (np.ndarray): The target position that the wolves are trying to reach.

    Returns:
        float: The fitness value (distance to the target).
    """
    return float(np.linalg.norm(position - target))


def update_position(wolf_pos: float, leader_pos: float, a: float) -> float:
    """
    Update the position of a wolf in relation to a leader wolf (alpha, beta, or gamma).

    Parameters:
        wolf_pos (float): The current position of the wolf in a specific dimension.
        leader_pos (float): The leader's position in that dimension.
        a (float): The decreasing parameter controlling exploration and exploitation.

    Returns:
        float: The new candidate position for the wolf in that dimension.
    """
    # Initialize the random number generator
    rng = np.random.default_rng(1)

    r1, r2 = rng.random(), rng.random()
    a = 2 * a * r1 - a
    c = 2 * r2
    d_leader = abs(c * leader_pos - wolf_pos)
    return leader_pos - a * d_leader


def update_wolves(
    wolves: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    gamma: np.ndarray,
    a: float,
    search_space: np.ndarray,
):
    """
    Update the positions of all wolves in the pack.

    Parameters:
        wolves (np.ndarray): Current positions of the wolves.
        alpha (np.ndarray): Position of the alpha wolf.
        beta (np.ndarray): Position of the beta wolf.
        gamma (np.ndarray): Position of the gamma wolf.
        a (float): The coefficient controlling the balance between exploration and exploitation.
        search_space (np.ndarray): The search space boundaries for each dimension.
    """
    for i, wolf in enumerate(wolves):
        for j in range(len(search_space)):
            # Compute new position based on alpha, beta, and gamma
            x1 = update_position(wolf[j], float(alpha[j]), a)
            x2 = update_position(wolf[j], float(beta[j]), a)
            x3 = update_position(wolf[j], float(gamma[j]), a)

            # Average the three positions and round to keep discrete values
            wolves[i, j] = int(round((x1 + x2 + x3) / 3))

            # Ensure position remains within bounds
            wolves[i, j] = np.clip(wolves[i, j], search_space[j, 0], search_space[j, 1])


def gwo_algorithm(
    search_space: np.ndarray, num_wolves: int, max_iterations: int
) -> np.ndarray:
    """
    Discrete Grey Wolf Optimizer (DGWO) algorithm for solving optimization problems.

    Parameters:
        search_space (np.ndarray): The search space boundaries for each dimension.
        num_wolves (int): The number of wolves in the pack.
        max_iterations (int): Maximum number of iterations to run the algorithm.

    Returns:
        np.ndarray: Position of the alpha wolf, representing the optimal solution found.
    """
    wolves = initialize_wolves(search_space, num_wolves)
    alpha, beta, gamma = (
        np.zeros(wolves.shape[1], dtype=int),
        np.zeros(wolves.shape[1], dtype=int),
        np.zeros(wolves.shape[1], dtype=int),
    )

    # Main optimization loop
    for iteration in range(max_iterations):
        a = 2 - (2 * iteration / max_iterations)

        # Update alpha, beta, and gamma wolves based on fitness
        for wolf in wolves:
            fitness = fitness_function(wolf)
            if fitness < fitness_function(alpha):
                alpha, beta, gamma = wolf.copy(), alpha.copy(), beta.copy()
            elif fitness < fitness_function(beta):
                beta, gamma = wolf.copy(), beta.copy()
            elif fitness < fitness_function(gamma):
                gamma = wolf.copy()

        # Update positions of wolves
        update_wolves(wolves, alpha, beta, gamma, a, search_space)

    return alpha


# Example usage
if __name__ == "__main__":
    # Define the discrete search space and parameters
    search_space_param = np.array([[-10, 10], [-10, 10]])  # Bounds for each dimension
    num_wolves_param = 10  # Number of wolves
    max_iterations_param = 100  # Number of iterations

    # Run the DGWO algorithm
    optimal_solution = gwo_algorithm(
        search_space_param, num_wolves_param, max_iterations_param
    )

    # Print the optimal solution and its fitness
    print("Optimal Solution:", optimal_solution)
    print("Fitness of Optimal Solution:", fitness_function(optimal_solution))
