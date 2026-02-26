'''multiple_regression_plane.py
multiple linear regression with two independent variable and demo of regression plane
Oliver W. Layton
CS 251/2: Data Analysis and Visualization
Spring 2026
'''

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
# for 3D plotting
from mpl_toolkits import mplot3d

# Use data jittered from surface z = 1 - 2x + 3y
x = np.arange(-4, 5)
y = np.array([4, -3, 5, -4, 1, -3, 4, -1, 3])
z = np.array([18.74, -1.1, 19.88, -5.71, 6.2, -10.37, 4.96, -5.3, 1.54])

# Convert x,y,z into Mx1 vectors
x = x[:, np.newaxis]
y = y[:, np.newaxis]
z = z[:, np.newaxis]

# Make a column vector of ones
oneCol = np.ones_like(x)

# Make A data matrix
A = np.hstack([oneCol, x, y])

# Solve Ac = z overdetermined system for c. Params 3x1 vector of: c0, c1, c2
c, res, rnk, s = sp.linalg.lstsq(A, z)

# Create x,y grid of sample points for plotting the plane
xSamp = np.linspace(np.min(x), np.max(x))
ySamp = np.linspace(np.min(y), np.max(y))
xPlane, yPlane = np.meshgrid(xSamp, ySamp)

# Evaluate equation for plane with solved for coefficients c0, c1, c2
fx = c[0, 0] + c[1, 0]*xPlane + c[2, 0]*yPlane

# Plot the plane and the data points
fig = plt.figure()
ax = plt.axes(projection='3d')

# Scatter plot of data
ax.scatter3D(x, y, z, color='blue')

# Surface plot of plane
ax.plot_surface(xPlane, yPlane, fx, alpha=0.9, color='black', edgecolor='none')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()