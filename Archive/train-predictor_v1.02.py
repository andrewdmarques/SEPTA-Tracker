
import os 
# Input variables
train_model = True
poi_heading = "E"


import pandas as pd
import re

# Change to the correcting working directory
dir_working = '/home/andrewdmarques/Desktop/Train-Tracker'
try:
    os.chdir(dir_working)
except FileNotFoundError:
    print(f"The directory {dir_working} does not exist.")

# Read the CSV file into a DataFrame
toi1 = pd.read_csv('toi.csv')

# Function to validate the rows and return cleaned dataframe and log
def clean_data(toi1):
    log = []
    valid_rows = []

    # Iterate through the rows and check conditions
    for index, row in toi1.iterrows():
        valid = True
        
        # Check if date_time is in the correct format 'yyyy-mm-dd hh:mm:ss'
        if not re.match(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", str(row['date_time'])):
            log.append(f"Row {index} removed: Invalid date_time format {row['date_time']}")
            valid = False

        # Check if train_heading is one of 'N', 'S', 'E', 'W'
        if row['train_heading'] not in ['N', 'S', 'E', 'W']:
            log.append(f"Row {index} removed: Invalid train_heading {row['train_heading']}")
            valid = False

        # Check if poi_dist is numeric
        if isinstance(row['poi_dist'], (int, float)) or str(row['poi_dist']).replace('.', '', 1).isdigit():
            # poi_dist is a valid number
            pass
        else:
            log.append(f"Row {index} removed: Invalid poi_dist {row['poi_dist']}")
            valid = False

        # Check if train_id is not blank
        if pd.isna(row['train_id']) or str(row['train_id']).strip() == "":
            log.append(f"Row {index} removed: Blank train_id")
            valid = False

        # Append valid row to valid_rows list
        if valid:
            valid_rows.append(row)

    # Create a new DataFrame with valid rows
    toi2 = pd.DataFrame(valid_rows)

    # Write log to a file
    with open('log_toi.txt', 'w') as log_file:
        log_file.write("\n".join(log))

    return toi2

# Apply the function to clean the data
toi2 = clean_data(toi1)

# Save the cleaned dataframe to a new CSV file
toi2.to_csv('toi2.csv', index=False)
print('Data cleaned')












import pandas as pd

# Function to add last1_train_heading, last1_poi_dist, last1_date_time, and train_heading_average efficiently
def add_last_train_info_and_heading_average_optimized(toi2):
    toi3 = toi2.copy()  # Create a copy of toi3 to avoid modifying the original DataFrame
    
    # Sort by train_id and date_time to ensure proper order for shifting
    toi3 = toi3.sort_values(by=['train_id', 'date_time'])
    
    # Use shift to create the previous records for each train_id group
    toi3['last1_train_heading'] = toi3.groupby('train_id')['train_heading'].shift(1)
    toi3['last1_poi_dist'] = toi3.groupby('train_id')['poi_dist'].shift(1)
    toi3['last1_date_time'] = toi3.groupby('train_id')['date_time'].shift(1)
    
    toi3['last2_train_heading'] = toi3.groupby('train_id')['train_heading'].shift(2)
    toi3['last2_poi_dist'] = toi3.groupby('train_id')['poi_dist'].shift(2)
    toi3['last2_date_time'] = toi3.groupby('train_id')['date_time'].shift(2)
    
    # Fill NaN values in last1 and last2 columns with the current row values
    toi3['last1_train_heading'].fillna(toi3['train_heading'], inplace=True)
    toi3['last1_poi_dist'].fillna(toi3['poi_dist'], inplace=True)
    toi3['last1_date_time'].fillna(toi3['date_time'], inplace=True)
    
    toi3['last2_train_heading'].fillna(toi3['train_heading'], inplace=True)
    toi3['last2_poi_dist'].fillna(toi3['poi_dist'], inplace=True)
    toi3['last2_date_time'].fillna(toi3['date_time'], inplace=True)
    
    # Apply mode function row-wise in a vectorized manner using a lambda function
    #toi3['train_heading_average'] = toi3[['train_heading', 'last1_train_heading', 'last2_train_heading']].mode(axis=1)[0]
    # This update it so that it looks across all train_id in the group not just the last 3 trains.
    toi3['train_heading_average'] = toi3.groupby('train_id')['train_heading'].transform(lambda x: x.mode()[0])
     
    return toi3

# Execute the optimized function to create toi3
toi3 = add_last_train_info_and_heading_average_optimized(toi2)

# Save the resulting DataFrame to CSV
toi3.to_csv('toi3.csv', index=False)

print('completed finding the previous train information')










# Only for training purposese

import pandas as pd
from datetime import datetime

# Function to create toi3 with the new time_to_poi and poi_time columns
def calculate_time_to_poi(toi3):
    toi4 = toi3.copy()  # Create a copy of toi3 to avoid modifying the original DataFrame
    toi4['date_time'] = pd.to_datetime(toi4['date_time'])  # Ensure date_time is in datetime format
    toi4['time_to_poi'] = float('nan')  # Initialize a new column for time_to_poi
    toi4['poi_time'] = pd.NaT  # Initialize a new column for poi_time
    
    # Iterate through each train_id
    for train_id, group in toi4.groupby('train_id'):
        # Find the row with the lowest poi_dist for the current train_id
        min_poi_dist_row = group.loc[group['poi_dist'].idxmin()]
        min_poi_time = min_poi_dist_row['date_time']
        
        # Update the poi_time column with the time corresponding to the lowest poi_dist
        toi4.loc[group.index, 'poi_time'] = min_poi_time
        
        # Calculate time difference in minutes and update time_to_poi column
        toi4.loc[group.index, 'time_to_poi'] = -1 * ((toi4.loc[group.index, 'date_time'] - min_poi_time).dt.total_seconds() / 60) # Times negative 1 so that positive values indicates minutes to arrival and negative value indicate time since the train has left that poi.

    return toi4

# Execute the function to create toi4
toi4 = calculate_time_to_poi(toi3)

toi4.to_csv('toi4.csv', index = False)

# Now make subset to just have the values that are approaching the station.
toi5 = toi4[(toi4['train_heading_average'] == poi_heading) & (toi4['time_to_poi'] >= 0)]
   
toi5.to_csv('toi5.csv', index=False)

print('filtered to have just the trains of interest')

