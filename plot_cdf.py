import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

sns.set_style("whitegrid")
df = pd.read_csv("./result/rsme.csv")

sns.kdeplot(data=df, x='rsme', hue="method", cumulative=True, common_norm=False)

plt.ylabel("CDF")
plt.xlabel("Error (m)")
plt.xlim([0, 5])
plt.tight_layout()
plt.savefig("./figures/rmse_cdf.png")
plt.show()

# errors = df['rsme']
#
# # Calculate ECDF
# values, base = np.histogram(errors, bins=500, density=True)
# cumulative = np.cumsum(values) / sum(values)  # Normalize cumulative values to get probabilities
#
# # Create DataFrame for estimated cumulative probabilities
# estimated_probabilities = pd.DataFrame({
#     'Error (m)': base[:-1],  # Exclude the last bin edge because np.histogram returns one more bin edge than bin
#     'Estimated Cumulative Probability': cumulative
# })
#
# # List of specific error values
# specific_errors = [0, 1, 2, 3, 4, 5]
#
# # Calculate and print the estimated cumulative probability for each specific error
# for error in specific_errors:
#     prob_error = estimated_probabilities[estimated_probabilities['Error (m)'] <= error]['Estimated Cumulative Probability'].max()
#     print(f"Estimated cumulative probability for error <= {error}m: {prob_error}")
