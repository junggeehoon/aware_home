import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
colors = plt.rcParams["axes.prop_cycle"]()

LABEL = 'L-02'

df = pd.read_csv('./data/sample.csv')
df = df.dropna()

data = df.loc[df['label'] == LABEL]
number_of_sensor = df.shape[1] - 3
fig = plt.figure(figsize=(15, 15))

for i in range(number_of_sensor):
    c1 = next(colors)["color"]
    c2 = next(colors)["color"]
    x = data.iloc[:, i].values
    print("mu, std for channel {}: {:.2f}, {:.2f}".format(i, np.mean(x), np.std(x)))
    plt.subplot(number_of_sensor, 1, i + 1)
    plt.plot(x, label='Channel ' + str(i), color=c1)
    plt.ylim([-100, -40])
    plt.ylabel('RSSI (dBm)')
    plt.legend(loc="upper right")

fig.tight_layout()
plt.savefig("./figures/rssi.png")
plt.show()
