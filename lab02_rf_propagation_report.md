# Laboratory 02: RF Propagation Modeling with Python

## Objective
Simulate Free-Space Path Loss (FSPL) to understand wireless coverage.

## Formula
FSPL(dB) = 20 log10(d) + 20 log10(f) + 32.44

Where:
- d = distance in kilometers
- f = frequency in MHz

## Parameters Used
- Frequency: 2400 MHz, which is 2.4 GHz WiFi
- Distance range: 0.1 km to 20 km

## Python Code
```python
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

```

## Sample Program Output
```text
RF Propagation Modeling with Python
Frequency: 2400 MHz

Distance (km)    FSPL (dB)
---------------------------
      0.10         80.04
      2.11        106.53
      4.12        112.34
      6.13        115.79
      8.14        118.26
     10.15        120.17
     12.16        121.74
     14.17        123.07
     16.18        124.22
     18.19        125.24
```

## Conclusion
The graph shows that free-space path loss increases as distance increases. At shorter distances, the signal loss is lower, but as the distance grows toward 20 km, the path loss becomes much higher. This means wireless coverage weakens as the receiver moves farther away from the transmitter.
