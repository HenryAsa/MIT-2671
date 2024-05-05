import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
from numpy.random import default_rng

# Creating "fake" data for the demo
a, b, c = -1, 30, -65
rng = default_rng(seed=314)  # Seed for random number generator

# Create parabolic data with 5 repeated "measurements" at each x value
x = np.repeat(np.arange(6, 28, 3), 5)
y = a*x**2 + b*x + c + 15 * rng.standard_normal(x.size)  # Add noise to the measurements

# Define the functional form of the fit
def model(x, a, b, c):
    return a*x**2 + b*x + c

# Execute the fit
popt, pcov = curve_fit(model, x, y, p0=[1, 1, 1])

# Compute the confidence interval for the fit parameters
alpha = 0.05  # 95% confidence interval -> 100*(1-alpha)
n = len(y)  # number of data points
p = len(popt)  # number of parameters
dof = max(0, n - p)  # degrees of freedom
# student-t value for the dof and confidence level
tval = stats.t.ppf(1.0-alpha/2., dof)

# New x variable over the domain
x_ = np.linspace(min(x), max(x), 100)

# Construct prediction interval of the function
y_pred = model(x_, *popt)
# Improved calculation of prediction interval
s_err = np.sqrt(np.sum((y - model(x, *popt))**2) / (n - p))
leverage = 1/n + (x_ - np.mean(x))**2 / np.sum((x - np.mean(x))**2)
ci = tval * s_err * np.sqrt(leverage)

y_upper = y_pred + ci
y_lower = y_pred - ci

# Plot results
plt.figure(figsize=(10, 6))
# Do shading first so it appears on the bottom
plt.fill_between(x_, y_lower, y_upper, color='magenta', alpha=0.2, edgecolor='none', label='95% confidence bounds')

# Plot the least squares fit of the data
plt.plot(x_, y_pred, 'b-', label='LSQ fit')
plt.plot(x, y, 'ro', label='Data')  # Plot data last so it appears on top

plt.xlabel('X (arb)')
plt.ylabel('Y (arb)')
plt.legend()
plt.show()
