import numpy as np

def generate_linear_system(n, filename="system.txt"):
    """
    Generates a random system of linear equations with 'n' variables.
    Ensures A is invertible and has no zero elements.
    """

    # Generate a random coefficient matrix (A) with values from -10 to 10, excluding 0
    while True:
        A = np.random.choice(list(range(-10, 0)) + list(range(1, 11)), (n, n)).astype(float)
        if np.linalg.det(A) != 0:  # Ensure A is invertible
            break

    # Generate a random solution vector (x_true)
    x_true = np.random.randint(-10, 11, n).astype(float)

    # Compute the right-hand side vector b
    b = np.dot(A, x_true)

    # Write equations to a file in C-compatible format
    with open(filename, "w") as f:
        f.write(f"// System of {n} linear equations\n\n")
        
        f.write("double A[][n] = {\n")
        for i in range(n):
            row = ", ".join([f"{A[i][j]:.2f}" for j in range(n)])
            f.write(f"    {{{row}}}")
            if i < n - 1:
                f.write(",\n")
            else:
                f.write("\n")
        f.write("};\n\n")
        
        f.write("double b[] = {")
        f.write(", ".join([f"{b[i]:.2f}" for i in range(n)]))
        f.write("};\n")

    print("Generated system saved to", filename)

    return A, b

# Usage Example
n = 10
A, b = generate_linear_system(n)
import numpy as np

def generate_linear_system(n, filename="system.txt"):
    """
    Generates a random system of linear equations with 'n' variables.
    Ensures A is invertible and has no zero elements.
    """

    # Generate a random coefficient matrix (A) with values from -10 to 10, excluding 0
    while True:
        A = np.random.choice(list(range(-10, 0)) + list(range(1, 11)), (n, n)).astype(float)
        if np.linalg.det(A) != 0:  # Ensure A is invertible
            break

    # Generate a random solution vector (x_true)
    x_true = np.random.randint(-10, 11, n).astype(float)

    # Compute the right-hand side vector b
    b = np.dot(A, x_true)

    # Write equations to a file in the required format
    with open(filename, "w") as f:
        f.write(f"# System of {n} linear equations\n")
        
        for i in range(n):
            row = " ".join([f"{A[i][j]:.2f}" for j in range(n)])
            f.write(row + "\n")
        
        f.write("\n")
        f.write(" ".join([f"{b[i]:.2f}" for i in range(n)]) + "\n")

    print("Generated system saved to", filename)
    
    return A, b

# Usage Example
n = 10
A, b = generate_linear_system(n)