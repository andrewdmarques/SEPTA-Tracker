#!/usr/bin/python3

# Description: The function of grouper is to make a csv that contains all of the trains of interest into one grouped csv.

# Set up libraries
import os
import time
import pandas as pd
from datetime import datetime
from math import radians, cos, sin, sqrt, atan2
from collections import Counter

##############################################################
# Prepare the environment
##############################################################
# Define the directory and file paths
dir_database = "Database"
dir_csv = os.path.join(dir_database, "Data_CSV")
toi_file = os.path.join(dir_database, "toi.csv")
train_line = "media"
poi_lat = 39.907821  # Add the point of interest for the latitude (best to use the train station of interest)
poi_lon = -75.328403  # Add the point of interest for the longitude (best to use the train station of interest)
poi_train_heading = "E"  # A string either N, S, E, or W that is a defining direction for the desired train

##############################################################
# Run
##############################################################

#!/usr/bin/python3

# Description: The function of grouper is to make a csv that contains all of the trains of interest into one grouped csv.

# Set up libraries
import os
import time
import pandas as pd
from datetime import datetime
from math import radians, cos, sin, sqrt, atan2

##############################################################
# Prepare the environment
##############################################################
# Define the directory and file paths
dir_database = "Database"
dir_csv = os.path.join(dir_database, "Data_CSV")
toi_file = os.path.join(dir_database, "toi.csv")
train_line = "media"
poi_lat = 39.907821  # Add the point of interest for the latitude (best to use the train station of interest)
poi_lon = -75.328403  # Add the point of interest for the longitude (best to use the train station of interest)
poi_train_heading = "E"  # A string either N, S, E, or W that is a defining direction for the desired train

##############################################################
# Run
##############################################################

#!/usr/bin/python3

# Description: The function of grouper is to make a csv that contains all of the trains of interest into one grouped csv.

# Set up libraries
import os
import time
import pandas as pd
from datetime import datetime
from math import radians, cos, sin, sqrt, atan2

##############################################################
# Prepare the environment
##############################################################
# Define the directory and file paths
dir_database = "Database"
dir_csv = os.path.join(dir_database, "Data_CSV")
toi_file = os.path.join(dir_database, "toi.csv")
train_line = "media"
poi_lat = 39.907821  # Add the point of interest for the latitude (best to use the train station of interest)
poi_lon = -75.328403  # Add the point of interest for the longitude (best to use the train station of interest)
poi_train_heading = "E"  # A string either N, S, E, or W that is a defining direction for the desired train

##############################################################
# Run
##############################################################


# Function to move old toi.csv file
def archive_old_toi_file():
    if os.path.exists(toi_file):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        new_name = os.path.join(dir_database, f"toi_before_{timestamp}.csv")
        os.rename(toi_file, new_name)

# Function to calculate distance using Haversine formula
def calculate_distance(lat1, lon1, lat2, lon2):
    R = 3958.8  # Radius of the Earth in miles
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    distance = R * c
    return distance

# Function to calculate speed in mph
def calculate_speed(distance, time_diff):
    # time_diff is in seconds, convert to hours
    time_diff_hours = time_diff / 3600.0
    if time_diff_hours == 0:
        return 0.0
    speed = distance / time_diff_hours
    return speed

# Function to check for new files and process them
def process_new_files():
    current_files = set(os.listdir(dir_csv))
    new_files = current_files - processed_files

    if new_files:
        for file in new_files:
            file_path = os.path.join(dir_csv, file)
            if os.path.isfile(file_path):
                df = pd.read_csv(file_path)
                if "line" in df.columns:
                    train_line_rows = df[df["line"].str.contains(train_line, case=False, na=False)].copy()  # Use .copy() to avoid the warning
                    if not train_line_rows.empty:
                        # Extract date and time from filename
                        filename_date_time = file.split('_')[1] + ' ' + file.split('_')[2].split('.')[0]
                        date_time_obj = datetime.strptime(filename_date_time, '%Y-%m-%d %H-%M-%S')
                        
                        train_line_rows.loc[:, 'filename'] = file
                        train_line_rows.loc[:, 'date'] = date_time_obj.strftime('%Y-%m-%d')
                        train_line_rows.loc[:, 'time'] = date_time_obj.strftime('%H:%M:%S')
                        train_line_rows.loc[:, 'date_time'] = date_time_obj
                        
                        # Calculate train_heading based on poi_train_heading and heading
                        if 'heading' in train_line_rows.columns:
                            train_line_rows = calculate_train_heading(train_line_rows, poi_train_heading)
                        
                        # Calculate poi_dist based on lat and lon
                        if 'lat' in train_line_rows.columns and 'lon' in train_line_rows.columns:
                            train_line_rows.loc[:, 'poi_dist'] = train_line_rows.apply(
                                lambda row: calculate_distance(poi_lat, poi_lon, row['lat'], row['lon']), axis=1)

                        # Create train_id column
                        train_line_rows['train_id'] = train_line_rows['date'] + "_" + train_line_rows['trainno'].astype(str)

                        # Add train_speed column with default value 0
                        train_line_rows['train_speed'] = 0.0

                        # Append or create the toi.csv file
                        if os.path.exists(toi_file):
                            toi_df = pd.read_csv(toi_file)
                            for idx, row in train_line_rows.iterrows():
                                train_id = row['train_id']
                                prev_entry = toi_df[toi_df['train_id'] == train_id].sort_values(by='date_time').iloc[-1] if not toi_df[toi_df['train_id'] == train_id].empty else None
                                if prev_entry is not None:
                                    prev_lat = prev_entry['lat']
                                    prev_lon = prev_entry['lon']
                                    prev_date_time = datetime.strptime(prev_entry['date_time'], '%Y-%m-%d %H:%M:%S')
                                    distance = calculate_distance(prev_lat, prev_lon, row['lat'], row['lon'])
                                    time_diff = (date_time_obj - prev_date_time).total_seconds()
                                    speed = calculate_speed(distance, time_diff)
                                    train_line_rows.at[idx, 'train_speed'] = float(speed)

                            train_line_rows.to_csv(toi_file, mode='a', header=False, index=False)
                        else:
                            train_line_rows.to_csv(toi_file, index=False)

        processed_files.update(new_files)

# Function to sort the toi.csv file and calculate train_heading_avg
def sort_and_group_toi_file():
    if os.path.exists(toi_file):
        df = pd.read_csv(toi_file)
        if 'filename' in df.columns and 'trainno' in df.columns:
            # Sort the DataFrame
            df = df.sort_values(by=['trainno', 'filename'])

            # Calculate train_heading_avg for each train_id group
            df['train_heading_avg'] = df.groupby('train_id')['train_heading'].transform(lambda x: Counter(x).most_common(1)[0][0])

            # Save the updated DataFrame to the file
            df.to_csv(toi_file, index=False)

# Function to calculate train_heading
def calculate_train_heading(df, poi_train_heading):
    if 'train_heading' not in df.columns:
        df['train_heading'] = None

    for idx, row in df.iterrows():
        if pd.isna(row['train_heading']):
            heading = row['heading']
            if poi_train_heading in ["E", "W"]:
                if 0 <= heading <= 180:
                    df.at[idx, 'train_heading'] = 'E'
                elif 180 < heading <= 360:
                    df.at[idx, 'train_heading'] = 'W'
            elif poi_train_heading in ["N", "S"]:
                if 0 <= heading <= 90 or 270 < heading <= 360:
                    df.at[idx, 'train_heading'] = 'N'
                elif 90 < heading <= 270:
                    df.at[idx, 'train_heading'] = 'S'
    return df

# Archive old toi.csv file before starting the loop
archive_old_toi_file()

# Set to keep track of processed files
processed_files = set()

# Main loop
while True:
    process_new_files()
    sort_and_group_toi_file()
    time.sleep(10)
