import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.preprocessing import MinMaxScaler
from scipy.ndimage import rotate
import seaborn as sns
import numpy as np
import pandas as pd
import pickle


labels = {
    'K-01': [0, 0],
    'K-02': [1, 0],
    'K-03': [2, 0],
    'K-04': [3, 0],
    'K-05': [1, 1],
    'K-06': [2, 1],
    'K-07': [3, 1],
    'K-08': [1, 2],
    'K-09': [2, 2],
    'K-10': [3, 2],
    'K-11': [0, 3],
    'K-12': [1, 3],
    'K-13': [2, 3],
    'H-01': [0, 4],
    'H-02': [1, 4],
    'H-03': [2, 4],
    'H-04': [3, 4],
    'H-05': [4, 4],
    'H-06': [5, 4],
    'H-07': [6, 4],
    'H-08': [7, 4],
    'H-09': [8, 4],
    'H-10': [9, 4],
    'H-11': [10, 4],
    'H-12': [11, 4],
    'L-01': [3, 5],
    'L-02': [4, 5],
    'L-03': [3, 6],
    'L-04': [4, 6],
    'L-05': [3, 7],
    'L-06': [2, 6],
    'L-07': [1, 7],
    'L-08': [2, 7],
    'L-09': [1, 8],
    'L-10': [2, 8]
}

floor_plan = plt.imread("./figures/floor_plan.png")
rotated = rotate(floor_plan, -90, reshape=True)

X = [-92,	-70,	-85,	-93,	-86,	-73,	-62,	-79,	-81,	-79,	-74,	-82]  # H-07

rf = pickle.load(open("./models/rf.pickle", "rb"))

probs = rf.predict_proba([X])

classes = rf.classes_

coordinates = [labels[label] for label in rf.classes_]

x_coords, y_coords = zip(*coordinates)
x = np.multiply(x_coords, 200)
y = np.multiply(y_coords, 200)

scaler = MinMaxScaler()
probs_norm = scaler.fit_transform(probs[0].reshape(-1, 1))

data = pd.DataFrame({
    'x': x,
    'y': y,
    'probability': probs_norm[:, 0]
})

# Assuming 'data' is your DataFrame
points = data[['x', 'y']].values
probabilities = data['probability'].values

# Define the kernel function
kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))

# Create and fit the Gaussian Process model
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
gp.fit(points, probabilities)

# Create a grid of points at which to evaluate the Gaussian Process
x = np.linspace(data['x'].min(), data['x'].max(), 100)
y = np.linspace(data['y'].min(), data['y'].max(), 100)
x_grid, y_grid = np.meshgrid(x, y)
xy = np.vstack([x_grid.ravel(), y_grid.ravel()]).T

# Use the GP to predict the function value at these points and the corresponding uncertainty
prob_pred, sigma = gp.predict(xy, return_std=True)

# Reshape for plotting
prob_grid = prob_pred.reshape(x_grid.shape)

# Plot the results
plt.figure()
plt.contourf(x_grid, y_grid, prob_grid, cmap='viridis')
plt.scatter(data['x'], data['y'], c=probabilities, cmap='viridis')
plt.xlabel('')
plt.ylabel('')
plt.xticks([])
plt.yticks([])
plt.tight_layout()
plt.show()

# diverging_colors = sns.color_palette("RdBu_r", 7)
#
# # plt.figure(figsize=(10,10))
# # plt.imshow(rotated)
# sns.scatterplot(data=data, x='x', y='y', hue='probability', palette=diverging_colors, legend=False)
#
# plt.xlabel('')
# plt.ylabel('')
# plt.xticks([])
# plt.yticks([])
# plt.tight_layout()
#
# plt.show()
