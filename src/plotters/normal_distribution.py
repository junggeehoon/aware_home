import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

df = pd.read_csv('./data/livingroom.csv')

data = df.iloc[:, 5].values
mu, std = norm.fit(data)

# Plot the histogram.
plt.hist(data, bins=30, density=True, alpha=0.6, color='b')

# Plot the PDF.
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)

plt.plot(x, p, 'k', linewidth=2)
title = "µ: {:.2f} σ: {:.2f}".format(mu, std)
plt.title(title)
plt.xlabel("RSSI (dBm)")

plt.show()
