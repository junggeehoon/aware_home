from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import pandas as pd
import numpy as np

# Prepare your data
train_data = pd.read_csv("./data/sample_small.csv")  # assuming that's your DataFrame
X_train = train_data[['rssi0', 'rssi1', 'rssi2', 'rssi3', 'rssi4', 'rssi5', 'rssi6', 'rssi7', 'rssi8', 'rssi9', 'rssi10', 'rssi11']]
y_train_x = train_data['x']
y_train_y = train_data['y']

# Define the kernel function
kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))

# Create and fit the Gaussian Process model for x-coordinate
gp_x = GaussianProcessRegressor(kernel=kernel)
gp_x.fit(X_train, y_train_x)

# Create and fit the Gaussian Process model for y-coordinate
gp_y = GaussianProcessRegressor(kernel=kernel)
gp_y.fit(X_train, y_train_y)

# Now, given new RSSI values (input_rssi), you can predict the (x, y) location.
input_rssi = np.array([[-92, -70, -85, -93, -86, -73, -62, -79, -81, -79, -74, -82]])
x_pred, sigma_x = gp_x.predict(input_rssi, return_std=True)
y_pred, sigma_y = gp_y.predict(input_rssi, return_std=True)

print(f"Predicted location is ({x_pred[0]}, {y_pred[0]}).")
