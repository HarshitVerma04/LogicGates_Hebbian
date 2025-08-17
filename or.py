import numpy as np
import matplotlib.pyplot as plt

# Bipolar truth table for OR
inputs = np.array([[+1, +1, 1],
                   [+1, -1, 1],
                   [-1, +1, 1],
                   [-1, -1, 1]])
targets = np.array([+1, +1, +1, -1])  # OR truth table (bipolar)

def hebbian_learning(inputs, targets, eta=1):
    w = np.zeros(inputs.shape[1])  # initialize weights
    weight_history = [w.copy()]
    for x, t in zip(inputs, targets):
        dw = eta * x * t
        w += dw
        weight_history.append(w.copy())
    return np.array(weight_history)

def plot_decision_boundary(inputs, targets, weights, title):
    plt.figure(figsize=(6,6))
    for x, t in zip(inputs, targets):
        if t == 1:
            plt.scatter(x[0], x[1], c="g", marker="o", s=100, label="Target +1" if "Target +1" not in plt.gca().get_legend_handles_labels()[1] else "")
        else:
            plt.scatter(x[0], x[1], c="r", marker="x", s=100, label="Target -1" if "Target -1" not in plt.gca().get_legend_handles_labels()[1] else "")
    w1, w2, b = weights
    x_vals = np.linspace(-2, 2, 100)
    if w2 != 0:
        y_vals = -(w1 * x_vals + b) / w2
        plt.plot(x_vals, y_vals, 'b-', label="Decision boundary")
    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.legend()
    plt.grid(True)
    plt.show()

# Train
weights = hebbian_learning(inputs, targets, eta=1)

# Plot weight updates
plt.figure(figsize=(7,5))
plt.plot(weights[:,0], label="w1")
plt.plot(weights[:,1], label="w2")
plt.plot(weights[:,2], label="bias")
plt.title("Hebbian Learning for OR - Weight Updates")
plt.xlabel("Update step")
plt.ylabel("Weight value")
plt.legend()
plt.grid(True)
plt.show()

print("Final weights for OR:", weights[-1])

# Plot decision boundary
plot_decision_boundary(inputs, targets, weights[-1], "OR Function Decision Boundary")
