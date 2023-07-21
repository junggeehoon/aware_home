import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import seaborn as sns
import pandas as pd

# Training the GPR model as you did before
rssi_scaler = StandardScaler()

df = pd.read_csv("../../data/sample_small.csv")
X = rssi_scaler.fit_transform(df.iloc[:, 0: -3])

coordinate_scaler = StandardScaler()
y = coordinate_scaler.fit_transform(df[['x', 'y']])

kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3)) \
    + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+1))

gpr = GaussianProcessRegressor(kernel=kernel, alpha=0.0).fit(X, y)

# Preparing the grid
x_grid = np.linspace(0, 10, 100)
y_grid = np.linspace(0, 8, 80)
x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)

# Generate input data for each point on the grid with fixed RSSI measurements
fixed_rssi = [-75, -85, -82, -92, -86, -87, -97, -82, -96, -85, -97, -101]  # Use your own fixed RSSI
input_data = []
for x, y in zip(x_mesh.ravel(), y_mesh.ravel()):
    input_data.append(fixed_rssi)

input_data_scaled = rssi_scaler.transform(input_data)  # Scale the data

# Predicting the likelihood for each point on the grid
mean, std = gpr.predict(input_data_scaled, return_std=True)

# De-standardize the predicted mean coordinates
mean = coordinate_scaler.inverse_transform(mean)

# Computing the likelihood (probability density) assuming a multivariate normal distribution
pdf = multivariate_normal.pdf(mean, mean=mean, cov=np.diag(std**2))

# Reshaping the probabilities to match the shape of the grid
pdf = pdf.reshape(x_mesh.shape)

# Plotting the probability density map
plt.figure(figsize=(8, 6))
plt.contourf(x_mesh, y_mesh, pdf, levels=100, cmap='viridis')
plt.colorbar(label='Probability Density')
plt.title('Location Probability Density')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.show()