import numpy as np
import pandas as pd
from pykalman import KalmanFilter

df = pd.read_csv('./data/sample.csv')

filtered_df = pd.DataFrame()

for label in df['label'].unique():
    df_label = df[df['label'] == label]
    print(label)
    filtered_df_label = pd.DataFrame()
    for column in df.columns:
        if 'rssi' in column:
            x = df_label[column].values
            
            # Initialize the Kalman filter
            kf = KalmanFilter(initial_state_mean=x[0], n_dim_obs=1)
            
            # Apply the Kalman filter
            (filtered_state_means, filtered_state_covariances) = kf.filter(x)
            
            # Add the filtered state means to the DataFrame
            filtered_df_label[column] = filtered_state_means.flatten()
        else:
            # If it's not an RSSI measurement, just copy the column
            filtered_df_label[column] = df_label[column].values
    
    # Append the filtered DataFrame for this label to the main DataFrame
    filtered_df = filtered_df.append(filtered_df_label, ignore_index=True)

# Save the filtered DataFrame to a new CSV file
filtered_df.to_csv('./data/kalman.csv', index=False)
