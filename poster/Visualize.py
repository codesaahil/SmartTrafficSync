import matplotlib.pyplot as plt
import numpy as np

# Provided data
lb = [0, 20, 40, 0, 20, 40]
ub = [20, 60, 100, 20, 60, 100]
best_params = np.array([13.22725542, 29.60728762, 62.89598097, 8.38153393, 36.05596045, 51.99602924])

# Infer initial parameters (e.g., midpoints of the bounds)
initial_params = np.array([(l + u) / 2 for l, u in zip(lb, ub)])

# Define the range for the x-axis (input variables)
x_range = np.linspace(0, 100, 200)

# Function to calculate the membership function value for a triangular MF
def triangular_mf(x, a, b, c):
    return np.maximum(0, (np.minimum((x - a) / (b - a), (c - x) / (c - b))))

# --- Plotting ---
fig, axs = plt.subplots(2, 1, figsize=(8, 10))  # Changed to 2 rows, 1 column
fig.suptitle("Fuzzy System Membership Functions: Initial vs. Optimized", fontsize=16)

# --- Plot for Input Variable A ---
axs[0].set_title("Input Variable A", fontsize=14)
axs[0].set_xlabel("Density", fontsize=12)
axs[0].set_ylabel("Membership Degree", fontsize=12)
axs[0].set_xlim(0, 100)
axs[0].set_ylim(0, 1.1)

# Initial state for Input A
axs[0].plot(x_range, triangular_mf(x_range, initial_params[0], initial_params[1], initial_params[2]),
             label='Initial - Low', linestyle='--')
axs[0].plot(x_range, triangular_mf(x_range, lb[0], initial_params[1], ub[1]),
             label='Initial - Medium', linestyle='--')
axs[0].plot(x_range, triangular_mf(x_range, initial_params[1], initial_params[2], ub[2]),
             label='Initial - High', linestyle='--')

# Best state for Input A
axs[0].plot(x_range, triangular_mf(x_range, lb[0], best_params[0], best_params[1]),
             label='Optimized - Low', color='blue')
axs[0].plot(x_range, triangular_mf(x_range, best_params[0], best_params[1], best_params[2]),
             label='Optimized - Medium', color='green')
axs[0].plot(x_range, triangular_mf(x_range, best_params[1], best_params[2], ub[2]),
             label='Optimized - High', color='red')

axs[0].legend()

# --- Plot for Input Variable B ---
axs[1].set_title("Input Variable B", fontsize=14)
axs[1].set_xlabel("Density", fontsize=12)
axs[1].set_ylabel("Membership Degree", fontsize=12)
axs[1].set_xlim(0, 100)
axs[1].set_ylim(0, 1.1)

# Initial state for Input B
axs[1].plot(x_range, triangular_mf(x_range, initial_params[3], initial_params[4], initial_params[5]),
             label='Initial - Low', linestyle='--')
axs[1].plot(x_range, triangular_mf(x_range, lb[3], initial_params[4], ub[4]),
             label='Initial - Medium', linestyle='--')
axs[1].plot(x_range, triangular_mf(x_range, initial_params[4], initial_params[5], ub[5]),
             label='Initial - High', linestyle='--')

# Best state for Input B
axs[1].plot(x_range, triangular_mf(x_range, lb[3], best_params[3], best_params[4]),
             label='Optimized - Low', color='blue')
axs[1].plot(x_range, triangular_mf(x_range, best_params[3], best_params[4], best_params[5]),
             label='Optimized - Medium', color='green')
axs[1].plot(x_range, triangular_mf(x_range, best_params[4], best_params[5], ub[5]),
             label='Optimized - High', color='red')

axs[1].legend()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to prevent title overlap
plt.savefig("fuzzy_optimization_top_bottom.png")
plt.show()