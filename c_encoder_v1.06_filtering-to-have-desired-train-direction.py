#!/usr/bin/python3
##########################################################################
# Description
##########################################################################
# encoder.py is intended to read in the "train of interest" file (toi.csv)
# and encode it into a matrix for machine learning.
# Data is encoded as the following:
# 1  lat [0-1] continuous latitude is limited to (for example) 39 to 41, where value is determined by (x-39)/2.
# 1  lon [0-1] continuous longitude is limited to (for example) -76 to -74, where the value is determined by (abs(x)-abs(-76))/2.
# 1  time [0-1] continuous time of day converted from 24 hours where 0 is 12:00am and 1 is 11:59pm.
# 7  day [matrix 0-1 using one-hot encoding] discrete for Sunday through Saturday.
# 1  time delay [0-1] continuous where 0 is 0 minutes and 1 is 100 minutes, where any value greater than 100 is rounded down to 100. 
# 1  speed [0-1] continuous hwere 0 is 0mph and 1 is 100mph where values greater than 100mph are reduced to 100mph.
# 40 trainno [matrix 0-1 using one-hot encoding] discrete showing the top 40 most common trains used for that line.
# 
# # Other variable not included in this scheme but could be added include:
# - service (local or express). Not included because for the stations of interest this does not affect train arrival time and this would be captured by trainno.
# - poi_distance (miles to station of interest). This is not included because the information should be captured by latitude and longitude. 
# - consist (the train cars that are presen in this train). This is not included because this information should be captured within trainno.

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

# Train and test groups.
train_group = 0.8 # values [0-1] for proportion of data points that should be used for training. The remaining (1-n) wll be the proportion of samples in the tesitng dataset.

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
    top_trainno["trainno"] = pd.to_numeric(top_trainno["trainno"])
    top_trainno = top_trainno.sort_values(by="trainno")
    top_trainno['index'] = range(1, trainno_max + 1)
    
    # Save the encoded information from the trainno for top 40 values.
    top_trainno.to_csv(file_encode_trainno, index=False)
    
 # Function that encodes the information from the train location to predict time.
def get_encoded(toi_e2, lat_max, lat_min, lon_max, lon_min, e_trainno):
    # Add the encoded columns.
    toi_e3 = toi_e2.copy()

    # Encode the lat and lon.
    toi_e3["encode_lat"] = (toi_e3["lat"] - lat_min) / (lat_max - lat_min)
    toi_e3["encode_lon"] = (toi_e3["lon"] - lon_min) / (lon_max - lon_min)

    # Encode the time.
    toi_e3['time'] = pd.to_datetime(toi_e3['time'], format='%H:%M')
    toi_e3['encode_time'] = toi_e3['time'].dt.hour * 3600 + toi_e3['time'].dt.minute * 60
    toi_e3['encode_time'] = toi_e3['encode_time'] / (23 * 3600 + 59 * 60)

    # Encode the time delay.
    max_late = 100  # value [0 to 999] for the number of minutes late that the train is running.
    toi_e3['encode_late'] = toi_e3['late'].apply(lambda x: min(max_late, x)) / max_late

    # Encode the train speed.
    max_speed = 100  # value [0 to 999] for the speed of the train in mph.
    toi_e3['encode_speed'] = toi_e3['train_speed'].apply(lambda x: min(max_speed, x)) / max_speed

    # Encode the day of the week and create one-hot encoding.
    toi_e3['date'] = pd.to_datetime(toi_e3['date'])
    days_of_week = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
    for day in days_of_week:
        toi_e3[f'encode_{day}'] = (toi_e3['date'].dt.day_name().str.lower() == day).astype(int)

    # Encode the train number.
    toi_e3['trainno'] = pd.to_numeric(toi_e3['trainno'], errors='coerce').fillna(0).astype(int)
    # Create a list of new column names based on unique trainno values
    e_trainno1 = "trainno_" + e_trainno['trainno'].astype(str)
    # Initialize all these new columns with 0s
    for col_name in e_trainno1:
        toi_e3[col_name] = 0
    # Loop through each row and set the appropriate column to 1
    for index, row in toi_e3.iterrows():
        column_name = "trainno_" + str(row['trainno'])
        if column_name in toi_e3.columns:
            toi_e3.at[index, column_name] = 1
    # Make sure that the trainno columns are clarified if they are encoding information for the NN.
    toi_e3.columns = toi_e3.columns.str.replace("trainno_", "encode_trainno_")

    return toi_e3   

# Function that takes in the data frame containing the encoded data and expected value and makes the numpy data frames for training and testing.    
def get_npy(df1,file_npy):
    df2 = df1.transpose()
    # Convert the dataframe to a NumPy array
    df2_array = df2.to_numpy()

    # Split the array into df2_array_x (all rows except the last) and df2_array_y (the last row)
    df2_array_x = df2_array[:-1, :]
    df2_array_y = df2_array[-1, :]
    df2_array_y = df2_array_y.reshape(1, -1)

    # Save the NumPy array as a .npy file
    file_npy_x = file_npy + '_x.npy'
    file_npy_y = file_npy + '_y.npy'
    np.save(file_npy_x, df2_array_x)
    np.save(file_npy_y, df2_array_y)    
    
    


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
    # Correctly create a copy of the dataframe
    toi_e1 = toi.copy()
    
    # Filter to just have the direction of travel that we want.
    toi_e1 = toi_e1[toi_e1['train_heading_avg'] == poi_train_heading]

    # Convert the 'date_time' and 'time_arrive' columns to datetime, handling mixed formats
    toi_e1['date_time'] = pd.to_datetime(toi_e1['date_time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')

    # Find the row with the minimum 'poi_dist' for each 'train_id'
    min_poi_dist = toi_e1.loc[toi_e1.groupby('train_id')['poi_dist'].idxmin()]

    # Create a dictionary to map each 'train_id' to its minimum 'time'
    time_arrive_map = min_poi_dist.set_index('train_id')['date_time'].to_dict()

    # Apply the mapping to create the 'time_arrive' column
    toi_e1['time_arrive'] = toi_e1['train_id'].map(time_arrive_map)

    # Convert the 'time' and 'time_arrive' columns to datetime
    toi_e1['time'] = pd.to_datetime(toi_e1['time'])
    toi_e1['time_arrive'] = pd.to_datetime(toi_e1['time_arrive'])

    # Calculate the difference in minutes between 'date_time' and 'time_arrive'
    toi_e1['time_to_arrive'] = (toi_e1['time_arrive'] - toi_e1['date_time']).dt.total_seconds() / 60.0

    # Filter out rows with negative 'time_to_arrive' values
    toi_e2 = toi_e1[toi_e1['time_to_arrive'] >= 0].copy()

    # Make the encoded version of 'time_to_arrive'
    max_time_to_arrive = 100
    toi_e2.loc[:, "value"] = np.minimum(1, toi_e2['time_to_arrive'] / max_time_to_arrive)


# Run the function to add columns with the encoded information.
toi_e3 = get_encoded(toi_e2, lat_max, lat_min, lon_max, lon_min, e_trainno)

# Create toi_e4 by filtering columns that contain the string "encode"
toi_e4 = toi_e3.filter(like="encode")

# Make the training and testing data sets.
if ml_training == True:
    toi_e5 = toi_e4.copy()
    # Determine the number of rows to be used for each of the differnet groups.
    nrow_tot = len(toi_e5)
    nrow_train = round(len(toi_e5)*train_group)
    nrow_test = nrow_tot - nrow_train

    # Add back the value column.
    toi_e5['value'] = toi_e2['value']

    # Randomly shuffle the rows.
    toi_e5['temp_index'] = np.random.permutation(np.arange(1, nrow_tot + 1))

    # Order based on the index.
    toi_e5 = toi_e5.sort_values(by='temp_index').reset_index(drop=True)
    
    # Remove the column used to randomize the rows.
    toi_e5 = toi_e5.drop(columns=['temp_index'])
    
    # Subset to have the training dataset which is the first section.
    toi_tr = toi_e5.iloc[:nrow_train]
    
    # Subset to have the testing dataset.
    toi_te = toi_e5.iloc[-nrow_test:]
    
    # Save the training dataset.
    file_npy_tr = 'septa02_tr'
    get_npy(toi_tr,file_npy_tr)
    
    # Save the testing dataset.
    file_npy_te = 'septa02_te'
    get_npy(toi_te,file_npy_te)
    
    



# Save to CSV
toi_e5.to_csv('temp2.csv', index=False)


