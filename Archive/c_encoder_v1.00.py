#!/usr/bin/python3
##########################################################################
# Description
##########################################################################
# encoder.py is intended to read in the "train of interest" file (toi.csv)
# and encode it into a matrix for machine learning.
# Data is encoded as the following:
lat [0-1] continuous
lon [0-1] continuous


trainno
##########################################################################
# Load libraries
##########################################################################

import csv
import pandas as pd

##########################################################################
# User input variables
##########################################################################

# These are the specified values for the latitude and longitude max and min and this will be to encode the train position as two columns with each between 0 and 1.
lat_max = 41
lat_min = 39
lon_max = -74
lon_min = -76


# Function that takes the toi data frame and makes a cssv file that will be used for one hot encoding 
def save_sorted_trainno_with_encode_position(toi, output_filename='sorted_trainno_with_encode_position.csv'):
 
    # Get the unique sorted train numbers
    unique_trainnos = sorted(toi['trainno'].unique())

    # Create a DataFrame with 'trainno' and 'encode_position'
    df_encoded = pd.DataFrame({
        'trainno': unique_trainnos,
        'encode_position': range(1, len(unique_trainnos) + 1)
    })

    # Add the 'other' row at the end
    df_encoded = df_encoded.append({'trainno': 'other', 'encode_position': len(unique_trainnos) + 1}, ignore_index=True)

    # Save the DataFrame to a CSV file
    df_encoded.to_csv(output_filename, index=False)
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








def encode_trainno_table(toi, file_toi):
    # Determine what the file name should be.
    file_encode_trainno = toi_file.replace(".csv", "_encode-trainno.csv")

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
    
    
