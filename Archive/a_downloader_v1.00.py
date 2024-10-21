 #!/usr/bin/python3
 
# Set up libraries
import os
import re
import json
import csv 
from datetime import datetime
import pandas as pd # This is for working with the files after the csv format 
import numpy as np # Used to calculate the distance from the poi to the train
import time # Used for 

##############################################################
# Prepare the environment
##############################################################
# Define the directory and file paths
dir_database = "Database"
dir_php = dir_database + "/Data_PHP"
dir_csv = dir_database + "/Data_CSV"
dir_data = "Data"
file_index = "index.php"
file_sch = dir_database + "/Schedule/train_schedule_2024-05-31.csv"
url_data = "https://www3.septa.org/api/TrainView/index.php"
train_line = "media"
poi_lat = 39.903810 # Add the point of interest for the latitude (best to use the train station of interest)
poi_lon = -75.335550 # Add the point of interest for the longitude (best to use the train station of interest)
poi_train_heading = "E" # A string either N, S, E, or W that is a defining direction for the desired train
hold_time = 6 # Time in minutes before updating the hold time. If 6 is selected, then after 3 minutes it will update to the schedule from 2.5 minutes ago.
file_dist_prev_hold = dir_database + "/distance_previous_hold.csv" # This is the last train directions from hold_time minutes ago divided by 2
file_dist_prev = dir_database + "/distance_previous.csv" # This is the last train directions from hold_time minutes ago
file_dist_curr = dir_database + "/current_distance.csv" # This is the train schedule with included distance from POI, showing the most recent data.

##############################################################
# Initialize environment.
##############################################################
# cd /mnt/d/Desktop/ADM/13_Projects/SEPTA-Tracker/Test-01/
sch1 = pd.read_csv(file_sch)
# Check if the directory exists, if not, create it
if not os.path.exists(dir_php):
    os.makedirs(dir_php)

if not os.path.exists(dir_csv):
    os.makedirs(dir_csv)

firt_time = True

###############################
# Download the train data and format it as as csv file
###############################

# Every 1 minute update the data
while True:
    # Download the train data
    os.system("rm -r index*")
    os.system(f"wget {url_data}")

    # Manage the downloaded data
    # Get the current date and time in the specified format
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Define the new file name with the current date and time
    file_php = f"index_{current_time}.php"
    file_php_path = os.path.join(dir_php, file_php)
    # Move the file to the new directory with the new name
    file_index = "index.php"
    os.system(f"mv {file_index} {file_php_path}")
    # Open the data 
    with open(file_php_path, 'r') as file:
        train0 = file.read()

    # Parse the JSON string into a dictionary
    train1 = json.loads(train0)
    # Save this as a csv file
    file_csv = f"index_{current_time}.csv"
    file_csv_path = os.path.join(dir_csv, file_csv)
    # Open a CSV file for writing
    with open(file_csv_path, 'w', newline='') as file:
        csv_writer = csv.writer(file)
        # Write the header (keys of the dictionary)
        header = train1[0].keys()  # Assuming the JSON is a list of dictionaries
        csv_writer.writerow(header)
        # Write the rows (values of the dictionaries)
        for item in train1:
            csv_writer.writerow(item.values())


    ###############################
    # Load in the train data
    ###############################

    # Load the train data file and the expected train file as a data frames
    train3 = pd.read_csv(file_csv_path)
    # Subset to only have the rows for the train line of interest
    train4 = train3[train3['line'].str.contains(train_line, case=False, na=False)].copy()
    train4.reset_index(drop=True, inplace=True) # Reset the index codes
    # Subset to only have the rows for the trains heading in the right direction

    # Ensure poi_train_heading is defined
    # Check if poi_train_heading is one of the acceptable values
    if poi_train_heading not in ['N', 'S', 'E', 'W']:
        raise ValueError("poi_train_heading is not one of the 4 acceptable values (N, S, E, W)")

    # Initialize the new column 'heading_direction' for all rows to 'None'
    train4['heading_direction'] = 'None'

    # Define the poi_train_heading variable (can be 'E', 'S', 'W', 'N')
    poi_train_heading = 'E'  # Example value

    # Loop through each row to assign 'heading_direction' based on 'heading'
    t_row = len(train4)
    for ii in range(t_row):
        val = train4['heading'].iloc[ii]
        if poi_train_heading == 'N' or poi_train_heading == 'S':
            if 90 <= val <= 269:
                train4.at[ii, 'heading_direction'] = 'S'
            elif 270 <= val <= 359:
                train4.at[ii, 'heading_direction'] = 'N'
            elif 0 <= val <= 89:
                train4.at[ii, 'heading_direction'] = 'N'
        
        if poi_train_heading == 'E' or poi_train_heading == 'W':
            if 0 <= val <= 179:
                train4.at[ii, 'heading_direction'] = 'E'
            elif 180 <= val <= 359:
                train4.at[ii, 'heading_direction'] = 'W'

        # Calculate the distance of trains to the poi 
        # Function to calculate Haversine distance
        def haversine(lat1, lon1, lat2, lon2):
            R = 3959.87433 # Radius of the Earth in miles
            phi1 = np.radians(lat1)
            phi2 = np.radians(lat2)
            delta_phi = np.radians(lat2 - lat1)
            delta_lambda = np.radians(lon2 - lon1)
            
            a = np.sin(delta_phi / 2.0) ** 2 + \
                np.cos(phi1) * np.cos(phi2) * \
                np.sin(delta_lambda / 2.0) ** 2
            
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
            
            return R * c

        # Calculate distance from the point of interest and add as a new column
        train4['poi_distance'] = train4.apply(
            lambda row: haversine(row['lat'], row['lon'], poi_lat, poi_lon), axis=1)

        # Save this as the current distance
        train4.to_csv(file_dist_curr, index=False)
        print(file_dist_curr)


        # Check if "previous_distance.csv" exists, if it doesn't then save the current file also as the previous file.
        if not os.path.exists(file_dist_prev):
            train4.to_csv(file_dist_prev, index=False)

        # Determine the distance traveled relative to the poi
        # Load the CSV files into DataFrames
        train5_p = pd.read_csv(file_dist_prev)
        train5_c = pd.read_csv(file_dist_curr)
        # Merge train5_c with train5_p on the trainno column
        train5_temp = train5_c.merge(train5_p[['trainno', 'poi_distance']], on='trainno', how='left', suffixes=('', '_prev'))
        # Create the dist_travel column
        train5_temp['dist_travel'] = train5_temp['poi_distance_prev'] - train5_temp['poi_distance']
        # Handle cases where there is no matching trainno
        train5_temp['dist_travel'] = train5_temp['dist_travel'].fillna(999)
        # Drop the previous poi_distance column
        train6 = train5_temp.drop(columns=['poi_distance_prev'])

        # Merge sch1 with train6 on the trainno column 
        train7 = sch1.merge(train6, on='trainno', how='right')
        #XXX Make this so that it merge

        # Subset to just have the trains that have not yet arrived (they have a positive distance).
        train7['poi_heading'] = train7['dist_travel'].apply(lambda x: 'away' if x < 0 else 'to')
        train7.to_csv("temp.csv", index=False)

        # If the current time is on the 3, then update the previous file.
        current_minute = datetime.now().minute
        # Logic: save an intermediate previous file that is ~half (rounded to the nearest whole minute) of what the specified hold time is. Make sure that it is not doing it more than once for each of the times that the current time is within the hold time (current_minute_previous logic), and if it is half then update using the last half hold file to become the new hold file and the current time becomes that intermediate half hold file for next time.
        hold_time = hold_time // 1 # Make sure that the hold time is always a whole number.
        hold_time_half = hold_time // 2
        if firt_time == True:
            first_time = False
            current_minute_last = current_minute - 1
        if current_minute % hold_time_half == 0:
            # Check that this is not the first time to run
            if current_minute != current_minute_last:
                # Check if file_dist_prev_hold.csv exists
                if os.path.exists(file_dist_prev_hold):
                    # Rename file_dist_prev_hold to file_dist_prev
                    os.rename(file_dist_prev_hold, file_dist_prev)
                else:
                    # Save train4 as file_dist_prev_hold
                    train4.to_csv(file_dist_prev_hold, index=False)
                current_minute_last = current_minute
    # Prepare to save a train-specific schedule
    # Subset just to have the trains with known heading directions (train8)
    train8 = train7[train7['heading_direction'].notna()]
    train8.reset_index(drop=True, inplace=True) # Reset the index codes

    # Create the 'train_file' column
    train8['train_file'] = "Database/Train_Files/trainno_" + train8['trainno'].astype(str) + ".csv"
    
    # Add the 'time_stamp' column filled with the current time string
    current_time_vector = [current_time] * len(train8)
    train8['time_stamp'] = current_time_vector

    # Iterate over each row in train8
    for index, row in train8.iterrows():
        file_path = row['train_file']
        
        # Check if the directory exists, if not, create it
        dir_name = os.path.dirname(file_path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        
        # Check if the file exists
        if os.path.isfile(file_path):
            # Append the row to the existing file
            row.to_frame().T.to_csv(file_path, mode='a', header=False, index=False)
        else:
            # Create the file and save the row
            row.to_frame().T.to_csv(file_path, mode='w', header=True, index=False)
    
    train8.to_csv("temp.csv", index=False)
    print("printed train 8*********************")
    print(list(train8))
    print(train8['trainno'])
    print(train8['train_file'])
    t_row = len(train8)
    for ii in range(t_row):
        print("\n-------\nValue Beloww")
        print(f"Value: {val}")
        print(f"trainno: {train8['trainno'].iloc[ii]}")
        print(f"heading_direction: {train8['heading_direction'].iloc[ii]}")
        print(f"poi_distance: {train8['poi_distance'].iloc[ii]}")
        print(f"dist_travel: {train8['dist_travel'].iloc[ii]}")
        print(f"poi_heading: {train8['poi_heading'].iloc[ii]}")


    # Filter train9 to include only rows where heading_direction equals poi_train_heading
    filtered_df = train8[train8['heading_direction'] == poi_train_heading]

    # Find the row with the minimum poi_distance
    min_distance_row = filtered_df.loc[filtered_df['poi_distance'].idxmin()]

    # Create train8 as a DataFrame containing just this row
    train9 = pd.DataFrame([min_distance_row])
    print(train9)

    time.sleep(30)
