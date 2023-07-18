import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style("whitegrid")

penguins = sns.load_dataset("penguins")
df = pd.DataFrame(penguins)


sns.relplot(x="flipper_length_mm", y="body_mass_g", data=df)
plt.show()