import pandas as pd
import numpy as np
from datetime import datetime


poi_train_heading = 'E'

################################################################################
# Functions
################################################################################

# Function to convert heading to cardinal direction based on specific rules
def convert_heading_to_cardinal(heading, poi_heading):
    if pd.isna(heading):
        return np.nan  # Keep NaN if heading is missing
    
    if poi_heading in ["E", "W"]:
        if 0 <= heading <= 180:
            return "E"
        elif 180 < heading <= 360:
            return "W"
    elif poi_heading in ["N", "S"]:
        if 0 <= heading <= 90 or 270 < heading <= 360:
            return "N"
        elif 90 < heading <= 270:
            return "S"
    
    return np.nan  # Return NaN if no condition is satisfied

# Function that determines the direction information and updates missing 'train_heading'
def get_dir(df, poi_train_heading):
    # Apply the logic to fill missing 'train_heading' values
    df['train_heading'] = df.apply(lambda row: convert_heading_to_cardinal(row['heading'], poi_train_heading) 
                                   if pd.isna(row['train_heading']) else row['train_heading'], axis=1)
    
    # Function to get the most common heading for each train_id
    def get_most_common_heading(heading_vector):
        return heading_vector.mode()[0] if not heading_vector.mode().empty else np.nan
    
    # Get the unique train IDs and calculate the most common heading for each train ID
    unique_train_ids = df['train_id'].unique()
    df['train_heading_avg'] = np.nan
    
    for train_id in unique_train_ids:
        rows = df[df['train_id'] == train_id]
        common_heading = get_most_common_heading(rows['train_heading'])
        df.loc[rows.index, 'train_heading_avg'] = common_heading
    
    return df

################################################################################
# Run
################################################################################

# Read in the specified data files.
con1 = pd.read_csv('config.csv')
# Load toi_file value from config
toi_file = '/media/andrewdmarques/Data01/Personal/Database/toi_14.csv'
df = pd.read_csv(toi_file)

# Convert the date_time column, handling inconsistent formats
df['date_time'] = pd.to_datetime(df['date_time'], format="%Y-%m-%d %H:%M:%S", errors='coerce')

# Handle missing or erroneous conversions due to format inconsistencies
df['date_time'] = pd.to_datetime(df['date_time'], format='mixed', errors='coerce')

# Convert the time column to just time
df['time'] = pd.to_datetime(df['time'], format="%H:%M:%S", errors='coerce').dt.time

# Subset to just have the trains of interest.
df2 = get_dir(df, poi_train_heading)
df2 = df2[df2['line'].str.contains('media', case=False)]
# Ensure 'train_heading_avg' doesn't contain NaN values when using .str.contains()
df2_clean = df2.dropna(subset=['train_heading_avg'])

# Now apply the str.contains method safely
df3 = df2_clean[df2_clean['train_heading_avg'].str.contains(poi_train_heading, case=False)]

# Remove any that are beyond 8 miles away, these tend to distort the data set
df3 = df3[df3['poi_dist'] < 8]

# Estimate when the time of arrival for the train would be.
df4 = df3.copy()
df4['time_arrival'] = df4['date_time']
df4['time_to_arrival'] = 0

for idx, row in df4.iterrows():
    t1 = row['train_id']
    t2 = df4[df4['train_id'] == t1]
    t3 = t2[t2['poi_dist'] == t2['poi_dist'].min()]
    
    df4.at[idx, 'time_arrival'] = t3['date_time'].iloc[0]
    
    time_diff = (t3['date_time'].iloc[0] - row['date_time']).total_seconds()
    df4.at[idx, 'time_to_arrival'] = time_diff / 60  # Convert seconds to minutes

# Remove any trains that were sitting for more than 1 hour
df5 = df4[df4['time_to_arrival'] < 60]

# Only look at the trains that have not yet arrived
df5 = df5[df5['time_to_arrival'] >= 0]

# Keep specific columns
df6 = df5[['lat', 'lon', 'time_to_arrival', 'train_id', 'poi_dist']]

# Model with speed unbounded ###################################
# Attempting to make a model that captures the train speed also
df7 = df5.copy()
df7['train_speed_t'] = df7['train_speed']

# Replace values in 'train_speed_t' that are below 10 with 20
df7.loc[df7['train_speed_t'] < 10, 'train_speed_t'] = 20
df7.loc[df7['train_speed_t'] > 20, 'train_speed_t'] = 20

# Create a linear model
from sklearn.linear_model import LinearRegression

X = df7[['train_speed_t', 'poi_dist']]
y = df7['time_to_arrival']
model = LinearRegression()
model.fit(X, y)

# Predict values and calculate error
df7['pred'] = model.predict(X)
df7['error'] = df7['time_to_arrival'] - df7['pred']

# Filter based on error
df_filtered = df7[(df7['error'] < 4) & (df7['error'] > -1)]
err_mod2 = len(df_filtered) / len(df7)

# Plotting
import matplotlib.pyplot as plt

plt.scatter(df7['poi_dist'], df7['time_to_arrival'], label="Data", alpha=0.5)
plt.plot(df7['poi_dist'], df7['pred'], color="red", label="Prediction")
plt.xlabel("POI Distance")
plt.ylabel("Time to Arrival (minutes)")
plt.title("POI Distance vs Time to Arrival with Regression Line")
plt.xlim(0, 8)
plt.legend()
plt.show()

# Add average speed over time (template for next steps)
df8 = df5.copy()
df8['speed_5'] = 0
for idx, row in df8.iterrows():
    tt = row['train_id']
    # Further processing if needed
