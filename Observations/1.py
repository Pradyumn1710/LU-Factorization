import matplotlib.pyplot as plt
import numpy as np

# Data from the image
cores = [1, 2, 4, 8, 16]
matrices = [100, 200, 300, 400]
computation_times = [
    [0.0173, 0.0344, 0.0422, 0.0598],
    [0.0113, 0.0153, 0.0257, 0.0336],
    [0.0053, 0.0093, 0.0165, 0.0202],
    [0.0023, 0.0059, 0.0104, 0.0124],
    [0.0033, 0.0052, 0.0074, 0.0099]
]

# Plot the data
plt.figure(figsize=(8, 6))
for i, matrix_size in enumerate(matrices):
    plt.plot(cores, [row[i] for row in computation_times], marker="o", label=f"{matrix_size} Matrices")

# Labels and title
plt.xlabel("Number of Cores")
plt.ylabel("Computation Time (seconds)")
plt.title("Computation Time vs. Number of Cores for Different Matrices")
plt.legend()
plt.grid()
plt.xticks(cores)

# Save the plot as a PNG file
plt.savefig("computation_time_plot.png", dpi=300)

# Show the plot
plt.show()