#!/usr/bin/python3
##########################################################################
# Description
##########################################################################
# encoder.py is intended to read in the "train of interest" file (toi.csv)
# and encode it into a matrix for machine learning.
# Data is encoded as the following:
1  lat [0-1] continuous latitude is limited to (for example) 39 to 41, where value is determined by (x-39)/2.
1  lon [0-1] continuous longitude is limited to (for example) -76 to -74, where the value is determined by (abs(x)-abs(-76))/2.
40 trainno [matrix 0-1 using one-hot encoding] discrete showing the top 40 most common trains used for that line.
1  time [0-1] continuous time of day converted from 24 hours where 0 is 12:00am and 1 is 11:59pm.
7  day [matrix 0-1 using one-hot encoding] discrete for Sunday through Saturday.
1  time delay [0-1] continuous where 0 is 0 minutes and 1 is 100 minutes, where any value greater than 100 is rounded down to 100. 
1  speed [0-1] continuous hwere 0 is 0mph and 1 is 100mph where values greater than 100mph are reduced to 100mph.

# Other variable not included in this scheme but could be added include:
- service (local or express). Not included because for the stations of interest this does not affect train arrival time and this would be captured by trainno.
- poi_distance (miles to station of interest). This is not included because the information should be captured by latitude and longitude. 
- consist (the train cars that are presen in this train). This is not included because this information should be captured within trainno.

# Output is values for


##########################################################################
# Load libraries
##########################################################################

import csv
import pandas as pd

##########################################################################
# User input variables
##########################################################################

# Make training/testing dataset.
ml_training = True # True/False statement, if True then it will make the values for time to train.  

# These are the specified values for the latitude and longitude max and min and this will be to encode the train position as two columns with each between 0 and 1.
lat_max = 41
lat_min = 39
lon_max = -74
lon_min = -76


##########################################################################
# Functions
##########################################################################

def encode_trainno_table(toi, file_encode_trainno):
    # Get the frequency of each trainno value
    frequency_table = toi['trainno'].value_counts().reset_index()

    # Rename columns for clarity
    frequency_table.columns = ['trainno', 'frequency']

    # Sort by frequency for better readability (optional)
    frequency_table = frequency_table.sort_values(by='frequency', ascending=False).reset_index(drop=True)

    # Step 1: Set trainno_max to 40
    trainno_max = 40

    # Step 2: Get the top 40 highest frequency rows
    top_trainno = frequency_table.nlargest(trainno_max, 'frequency')

    # Step 3: If there are fewer than 40 rows, add additional "other" rows
    num_rows_to_add = trainno_max - len(top_trainno)
    if num_rows_to_add > 0:
        additional_rows = pd.DataFrame({
            'trainno': ['0'] * num_rows_to_add,
            'frequency': [0] * num_rows_to_add
        })
        top_trainno = pd.concat([top_trainno, additional_rows], ignore_index=True)

    # Step 4: Create a new 'index' column with values from 1 to 40
    top_trainno['index'] = range(1, trainno_max + 1)
    
    # Save the encoded information from the trainno for top 40 values.
    top_trainno.to_csv(file_encode_trainno, index=False)
    
    


##########################################################################
# Inititalization
##########################################################################

# Open the configuration file and read the data
config_file = '/media/andrewdmarques/Data01/Personal/config.csv'
with open(config_file) as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['variable'] == 'dir_database':
            dir_database = row['value']
        elif row['variable'] == 'dir_csv':
            dir_csv = row['value']
        elif row['variable'] == 'toi_file':
            toi_file = row['value']
        elif row['variable'] == 'train_line':
            train_line = row['value']
        elif row['variable'] == 'poi_lat':
            poi_lat = float(row['value'])
        elif row['variable'] == 'poi_lon':
            poi_lon = float(row['value'])
        elif row['variable'] == 'poi_train_heading':
            poi_train_heading = row['value']
        elif row['variable'] == 'dir_php':
            dir_php = row['value']
        elif row['variable'] == 'url_data':
            url_data = row['value']
        elif row['variable'] == 'scan_count':
            scan_count = float(row['value'])
            
##############################################################
# Run
##############################################################

# Read in the toi file.
toi = pd.read_csv(toi_file)

# Determine which trainno should be encoded.
file_encode_trainno = toi_file.replace(".csv", "_encode-trainno.csv")
encode_trainno_table(toi, file_encode_trainno)

# Read in the encoding trainno
e_trainno = pd.read_csv(file_encode_trainno)

# Begin encoding the trains as an item of length 52 items long.
if ml_training == True:
    
toi_e1 = toi
# Find the row with the minimum 'poi_dist' for each 'train_id'
min_poi_dist = toi_e1.loc[toi_e1.groupby('train_id')['poi_dist'].idxmin()]

# Create a dictionary to map each 'train_id' to its minimum 'time'
time_arrive_map = min_poi_dist.set_index('train_id')['date'].to_dict()

# Apply the mapping to create the 'time_arrive' column
toi_e1['time_arrive'] = toi_e1['train_id'].map(time_arrive_map)

# Convert the 'time' and 'time_arrive' columns to datetime
toi_e1['time'] = pd.to_datetime(toi_e1['time'])
toi_e1['time_arrive'] = pd.to_datetime(toi_e1['time_arrive'])

# Calculate the difference in minutes between 'time' and 'time_arrive'
toi_e1['time_to_arrive'] = (toi_e1['time_arrive'] - toi_e1['time']).dt.total_seconds() / 60.0
toi_e1.to_csv('temp1.csv', index=False)


