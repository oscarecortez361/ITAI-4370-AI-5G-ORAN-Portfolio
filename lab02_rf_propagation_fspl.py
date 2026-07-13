import numpy as np
import matplotlib.pyplot as plt

# Parameters
frequency = 2400  # MHz (2.4 GHz WiFi)
distances = np.linspace(0.1, 20, 100)  # km

# FSPL calculation
fspl = 20 * np.log10(distances) + 20 * np.log10(frequency) + 32.44

# Print sample output values
print("RF Propagation Modeling with Python")
print("Frequency:", frequency, "MHz")
print()
print("Distance (km)    FSPL (dB)")
print("---------------------------")

for d, loss in zip(distances[::10], fspl[::10]):
    print(f"{d:10.2f}      {loss:8.2f}")

# Plot
plt.figure(figsize=(8, 5))
plt.plot(distances, fspl, label="FSPL at 2.4 GHz")
plt.xlabel("Distance (km)")
plt.ylabel("Path Loss (dB)")
plt.title("Free-Space Path Loss vs Distance")
plt.grid(True)
plt.legend()
plt.show()
