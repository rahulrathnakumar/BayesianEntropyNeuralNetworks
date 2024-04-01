import numpy as np
import matplotlib.pyplot as plt

# load data from S2_016.txt
data = np.loadtxt('S2_016.txt')


# Parsing the data into two separate lists for x and y values
x_values = []
y_values = []

for line in data:
    x, y = line[0], line[1]
    x_values.append(float(x))
    y_values.append(float(y))

# Converting lists to numpy arrays for easier manipulation
x_values = np.array(x_values)
y_values = np.array(y_values)

# Introducing sinusoidal fluctuations after x = 10.0
# The frequency and amplitude of the sinusoid can be adjusted as needed
x_mod = x_values[x_values >= 10]
y_mod = y_values[x_values >= 10]
amplitude = 0.009  # Example amplitude
frequency = 0.4   # Example frequency

# Creating the sinusoidal function
y_mod_sin = y_mod.mean() + amplitude * np.sin(2 * np.pi * frequency * (x_mod - 10))

# Replacing the original y values with the modified values
y_values[x_values >= 10] = y_mod_sin

# Plotting the original and modified two point correlation function
plt.figure(figsize=(12, 6))
plt.plot(x_values, y_values, label='Modified', color='blue')
plt.axvline(x=10, color='red', linestyle='--', label='x = 10.0')
plt.xlabel('X')
plt.ylabel('Two Point Correlation Function')
plt.title('Modified Two Point Correlation Function with Sinusoidal Fluctuations')
plt.legend()
plt.grid(True)
plt.savefig('modified_tpc.png')

# save modified data to S2_016_modified.txt
np.savetxt('S2_016_modified.txt', np.transpose([x_values, y_values]))

