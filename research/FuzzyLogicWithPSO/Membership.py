import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz

def plot_membership_functions():
    # Create universe of discourse
    density = np.arange(0, 101, 1)
    duration = np.arange(10, 61, 1)

    # Density Membership Functions (Example parameters)
    low_density = fuzz.trapmf(density, [0, 0, 20, 40])
    medium_density = fuzz.trimf(density, [20, 50, 80])
    high_density = fuzz.trapmf(density, [60, 80, 100, 100])

    # Duration Membership Functions
    short_duration = fuzz.trimf(duration, [10, 20, 30])
    medium_duration = fuzz.trimf(duration, [20, 40, 50])
    long_duration = fuzz.trimf(duration, [40, 50, 60])

    # Plotting
    plt.figure(figsize=(15, 10))

    # Density Membership Functions
    plt.subplot(2, 1, 1)
    plt.plot(density, low_density, 'b', label='Low Density')
    plt.plot(density, medium_density, 'g', label='Medium Density')
    plt.plot(density, high_density, 'r', label='High Density')
    plt.title('Traffic Density Membership Functions')
    plt.xlabel('Density (%)')
    plt.ylabel('Membership Degree')
    plt.legend()
    plt.grid(True)

    # Duration Membership Functions
    plt.subplot(2, 1, 2)
    plt.plot(duration, short_duration, 'b', label='Short Duration')
    plt.plot(duration, medium_duration, 'g', label='Medium Duration')
    plt.plot(duration, long_duration, 'r', label='Long Duration')
    plt.title('Traffic Light Duration Membership Functions')
    plt.xlabel('Duration (seconds)')
    plt.ylabel('Membership Degree')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('membership_functions.png')
    plt.close()

# Generate the visualization
plot_membership_functions()

# Demonstrative printing to show the visualization was created
print("Membership function visualization has been saved as 'membership_functions.png'")

# Additional function to show fuzzy set calculations
def demonstrate_fuzzy_inference():
    # Example traffic scenario
    density_values = [25, 60, 85]  # Low, Medium, High density scenarios
    
    # Density Membership Functions
    low_density = fuzz.trapmf(np.arange(0, 101, 1), [0, 0, 20, 40])
    medium_density = fuzz.trimf(np.arange(0, 101, 1), [20, 50, 80])
    high_density = fuzz.trapmf(np.arange(0, 101, 1), [60, 80, 100, 100])
    
    print("\nFuzzy Set Membership Calculations:")
    for value in density_values:
        low_membership = fuzz.interp_membership(np.arange(0, 101, 1), low_density, value)
        medium_membership = fuzz.interp_membership(np.arange(0, 101, 1), medium_density, value)
        high_membership = fuzz.interp_membership(np.arange(0, 101, 1), high_density, value)
        
        print(f"\nDensity Value: {value}")
        print(f"Low Density Membership:      {low_membership:.2f}")
        print(f"Medium Density Membership:   {medium_membership:.2f}")
        print(f"High Density Membership:     {high_membership:.2f}")

# Run the demonstrative fuzzy inference
demonstrate_fuzzy_inference()