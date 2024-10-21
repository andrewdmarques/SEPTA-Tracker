
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import re
from datetime import datetime
import pickle
import numpy as np


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

############################################################
# Input variables
############################################################

poi_heading = "E"
# Change to the correcting working directory
dir_working = '/home/andrewdmarques/Desktop/Train-Tracker'
file_in = 'toi_v2.csv' # File that has the data to be trained and/or predicted with
file_model = 'model.pkl' # The file that has the trained decision tree
file_poly = 'poly_transformer.pkl' # The file that contains the polynomial fit for the data.
model_train = False
model_predict = True

############################################################
# Functions
############################################################

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

# Function to add last1_train_heading, last1_poi_dist, last1_date_time, and train_heading_average efficiently
def add_last_train_info_and_heading_average_optimized(toi2):
    toi3 = toi2.copy()  # Create a copy of toi3 to avoid modifying the original DataFrame
    
    # Sort by train_id and date_time to ensure proper order for shifting
    toi3 = toi3.sort_values(by=['train_id', 'date_time'])
        
    # Use shift to create the previous records for each train_id group
    toi3['last1_train_heading'] = toi3.groupby('train_id')['train_heading'].shift(1)
    toi3['last1_poi_dist'] = toi3.groupby('train_id')['poi_dist'].shift(1)
    toi3['last1_date_time'] = toi3.groupby('train_id')['date_time'].shift(1)
    toi3['last1_train_speed'] = toi3.groupby('train_id')['train_speed'].shift(1)

    toi3['last2_train_heading'] = toi3.groupby('train_id')['train_heading'].shift(2)
    toi3['last2_poi_dist'] = toi3.groupby('train_id')['poi_dist'].shift(2)
    toi3['last2_date_time'] = toi3.groupby('train_id')['date_time'].shift(2)
    toi3['last2_train_speed'] = toi3.groupby('train_id')['train_speed'].shift(2)

    toi3['last3_train_heading'] = toi3.groupby('train_id')['train_heading'].shift(3)
    toi3['last3_poi_dist'] = toi3.groupby('train_id')['poi_dist'].shift(3)
    toi3['last3_date_time'] = toi3.groupby('train_id')['date_time'].shift(3)
    toi3['last3_train_speed'] = toi3.groupby('train_id')['train_speed'].shift(3)

    toi3['last4_train_heading'] = toi3.groupby('train_id')['train_heading'].shift(4)  # Added for last4
    toi3['last4_poi_dist'] = toi3.groupby('train_id')['poi_dist'].shift(4)  # Added for last4
    toi3['last4_date_time'] = toi3.groupby('train_id')['date_time'].shift(4)  # Added for last4
    toi3['last4_train_speed'] = toi3.groupby('train_id')['train_speed'].shift(4)  # Added for last4

    # Fill NaN values in last1, last2, last3, and last4 columns with the current row values
    toi3['last1_train_heading'] = toi3['last1_train_heading'].fillna(toi3['train_heading'])
    toi3['last1_poi_dist'] = toi3['last1_poi_dist'].fillna(toi3['poi_dist'])
    toi3['last1_date_time'] = toi3['last1_date_time'].fillna(toi3['date_time'])
    toi3['last1_train_speed'] = toi3['last1_train_speed'].fillna(toi3['train_speed'])

    toi3['last2_train_heading'] = toi3['last2_train_heading'].fillna(toi3['train_heading'])
    toi3['last2_poi_dist'] = toi3['last2_poi_dist'].fillna(toi3['poi_dist'])
    toi3['last2_date_time'] = toi3['last2_date_time'].fillna(toi3['date_time'])
    toi3['last2_train_speed'] = toi3['last2_train_speed'].fillna(toi3['train_speed'])

    toi3['last3_train_heading'] = toi3['last3_train_heading'].fillna(toi3['train_heading'])
    toi3['last3_poi_dist'] = toi3['last3_poi_dist'].fillna(toi3['poi_dist'])
    toi3['last3_date_time'] = toi3['last3_date_time'].fillna(toi3['date_time'])
    toi3['last3_train_speed'] = toi3['last3_train_speed'].fillna(toi3['train_speed'])

    toi3['last4_train_heading'] = toi3['last4_train_heading'].fillna(toi3['train_heading'])
    toi3['last4_poi_dist'] = toi3['last4_poi_dist'].fillna(toi3['poi_dist'])
    toi3['last4_date_time'] = toi3['last4_date_time'].fillna(toi3['date_time'])
    toi3['last4_train_speed'] = toi3['last4_train_speed'].fillna(toi3['train_speed'])


    # Apply mode function row-wise in a vectorized manner using a lambda function
    #toi3['train_heading_average'] = toi3[['train_heading', 'last1_train_heading', 'last2_train_heading']].mode(axis=1)[0]
    # This update it so that it looks across all train_id in the group not just the last 3 trains.
    toi3['train_heading_average'] = toi3.groupby('train_id')['train_heading'].transform(lambda x: x.mode()[0])
     
    return toi3

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

############################################################
# Run
############################################################

# Ensure that the file structue exists.
try:
    os.chdir(dir_working)
except FileNotFoundError:
    print(f"The directory {dir_working} does not exist.")

######################
# Open and clean the data
toi1 = pd.read_csv(file_in)

# Apply the function to clean the data
toi2 = clean_data(toi1)

# Save the cleaned dataframe to a new CSV file
toi2.to_csv('toi2.csv', index=False)
print('Data cleaned')

######################
# Get the previous train data if it exists
toi3 = add_last_train_info_and_heading_average_optimized(toi2)

# Save the resulting DataFrame to CSV
toi3.to_csv('toi3.csv', index=False)

print('completed finding the previous train information')

######################
if model_train == True:
    # Prepare the data so that it can be used for training the model.
    # Execute the function to create toi4
    toi4 = calculate_time_to_poi(toi3)

    toi4.to_csv('toi4.csv', index = False)
    
    # Now make subset to just have the values that are approaching the station.
    toi5 = toi4[(toi4['train_heading_average'] == poi_heading) & (toi4['time_to_poi'] >= 0)]
       
    toi5.to_csv('toi5.csv', index=False)

if model_train == False:
    # Filtering based on making predictions only.
    toi4 = toi3.copy()
    toi5 = toi4[(toi4['train_heading_average'] == poi_heading)]
    
    # Read in the csv file to assign the variables.   
    try:
        # Read the CSV file
        df = pd.read_csv('lat_lon.csv')
        # Assign variables based on the contents of the CSV
        lat_max = df.loc[df['variable'] == 'lat_max', 'value'].values[0]
        lat_min = df.loc[df['variable'] == 'lat_min', 'value'].values[0]
        lon_max = df.loc[df['variable'] == 'lon_max', 'value'].values[0]
        lon_min = df.loc[df['variable'] == 'lon_min', 'value'].values[0]
        # Print the values to confirm
        print(f"lat_max: {lat_max}, lat_min: {lat_min}, lon_max: {lon_max}, lon_min: {lon_min}")
    except Exception as e:
        print("Failed to assign variables from CSV file. Error:", e)

    # use the lat and lon max and min values to filter the data frame.
    # Define the filtering conditions
    lat_condition = (toi5['lat'] >= lat_min) & (toi5['lat'] <= lat_max)
    lon_condition = (toi5['lon'] >= lon_min) & (toi5['lon'] <= lon_max)

    # Apply the conditions to filter the DataFrame
    toi7 = toi5[lat_condition & lon_condition]
    toi7.to_csv('toi7.csv', index=False)
    
   


print('filtered to have just the trains of interest')

######################

if model_train == True: # If the linear regression is to be trained by the data.
    print('Beginning Training')

    # Prepare the training data further.
    # Filter to have just the cleanest trains -- removing those trains that take longer than expected.
    # Load the dataset
    toi6 = pd.read_csv('toi5.csv')

    # Identify the train_ids that have any row with time_to_poi greater than 30 or greater than 8
    train_ids_to_remove = toi6[(toi6['time_to_poi'] > 30) | (toi6['poi_dist'] > 8)]['train_id'].unique()

    # Filter the dataset to remove rows with those train_ids
    filtered_toi6 = toi6[~toi6['train_id'].isin(train_ids_to_remove)]
    
    # Filter to remove train groups that do not really make it to the station.
    filtered_toi6 = filtered_toi6.groupby('train_id').filter(lambda x: x['poi_dist'].min() <= 0.25)

    # If you want to save it to a new CSV file
    filtered_toi6.to_csv('toi6.csv', index=False)
    

    ######################
    # Train the model and save it.
    # Load the dataset
    toi7 = pd.read_csv('toi6.csv')
    
    # Record what the min and max lat and lon are for the samples.
    lat_max = max(toi7['lat'])
    lat_min = min(toi7['lat'])
    lon_max = max(toi7['lon'])
    lon_min = min(toi7['lon'])
    
    # Save the variables to the csv file.
    # Create a dictionary with variable names and their values
    data = {
        'variable': ['lat_max', 'lat_min', 'lon_max', 'lon_min'],
        'value': [lat_max, lat_min, lon_max, lon_min]
    }
    # Convert the dictionary into a DataFrame
    df = pd.DataFrame(data)
    # Save the DataFrame to a CSV file
    df.to_csv('lat_lon.csv', index=False)
  
    # Independent variables including past distances
    X = toi7[['poi_dist', 'last1_poi_dist', 'last2_poi_dist','last3_poi_dist','last4_poi_dist','train_speed','last1_train_speed','last2_train_speed','last3_train_speed','last4_train_speed']]
    y = toi7['time_to_poi']  # Dependent variable

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Optional: Apply polynomial features if you want to capture polynomial non-linearities
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    
    # Save the fitted polynomial.
    # Save the fitted PolynomialFeatures to a file
    with open(file_poly, 'wb') as f:
        pickle.dump(poly, f)

    # Initialize and fit a Random Forest model to capture non-linear relationships
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_poly, y_train)

    # Make predictions
    y_pred = model.predict(X_test_poly)



    # Save the trained model using pickle
    with open(file_model, 'wb') as model_file:
        pickle.dump(model, model_file)
    


    # Convert to DataFrame for easier filtering and manipulation
    results_df = pd.DataFrame({'actual': y_test, 'predicted': y_pred})

    # Filter the data to include only rows where actual time_to_poi is between 12 and 17
    filtered_df = results_df[(results_df['actual'] >= 12) & (results_df['actual'] <= 17)]

    # Calculate the difference between predicted and actual values (errors)
    filtered_df['error'] = filtered_df['predicted'] - filtered_df['actual']

    # Calculate the mean and standard deviation of the errors
    mean_error = np.mean(filtered_df['error'])
    std_error = np.std(filtered_df['error'])
    two_std = 2 * std_error  # Two standard deviations

    # Create a histogram of the errors with percentage on the y-axis
    plt.figure(figsize=(10, 6))
    plt.hist(filtered_df['error'], bins=20, color='skyblue', edgecolor='black', density=True)

    # Convert the y-axis from density to percentage
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y * 100:.0f}%'))

    # Add titles and labels, including two standard deviation values in the title
    plt.title(f'Histogram of Prediction Errors (Actual time_to_poi between 12 and 17 minutes)\n'
              f'Mean Error: {mean_error:.2f}, 2 Std Dev: ±{two_std:.2f}')
    plt.xlabel('Prediction Error (Predicted - Actual)')
    plt.ylabel('Percentage')

    # Show the plot
    plt.show()
    
    # Save the model predictions.
    results_df.to_csv('prediction.csv',index = False)


if model_predict == True and model_train == False:
    # To load the model for future predictions
    with open(file_model, 'rb') as model_file:
        loaded_model = pickle.load(model_file)
        
        # Prepare the files to make the predictions.
        X = toi7[['poi_dist', 'last1_poi_dist', 'last2_poi_dist', 'last3_poi_dist', 'last4_poi_dist', 
                  'train_speed', 'last1_train_speed', 'last2_train_speed', 'last3_train_speed', 'last4_train_speed']]
        
        # Open the previously trained polynomial fit for the data.
        # Load the saved PolynomialFeatures transformer
        with open(file_poly, 'rb') as f:
            poly = pickle.load(f)
        
        # Prepare the X values for prediction.
        X_poly = poly.transform(X)
        y_pred = loaded_model.predict(X_poly)
        
        # Convert to DataFrame for easier filtering and manipulation
        results_df = pd.DataFrame({
            'date': toi7['date'],
            'time': toi7['time'],
            'train_id': toi7['train_id'],
            'train_heading_average': toi7['train_heading_average'],
            'poi_dist': toi7['poi_dist'],
            'lat': toi7['lat'],
            'lon': toi7['lon'],
            'predicted': y_pred
        })

        # Save the model predictions.
        results_df.to_csv('prediction_no-model.csv', index=False)
        print('Completed making predictions')
        
        # Make a plot that shows the prediction based on the location
        # Create the scatter plot
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(results_df['lon'], results_df['lat'], c=results_df['predicted'], cmap='viridis', alpha=0.7)
        plt.colorbar(scatter, label='Predicted')
        # Set axis labels and plot title
        plt.ylabel('Latitude')
        plt.xlabel('Longitude')
        plt.title('Latitude and Longitude Colored by Predicted Values')
        # Display the plot
        plt.show()
        
        # Determine the train that is most close to the station.
        # Step 1: Sort by 'date' and 'time' to get the most recent entry
        results_df_sorted = results_df.sort_values(by=['date', 'time'], ascending=[False, False])

        # Step 2: Filter to get the most recent records based on date and time
        most_recent_records = results_df_sorted[results_df_sorted['date'] == results_df_sorted.iloc[0]['date']]
        most_recent_records = most_recent_records[most_recent_records['time'] == results_df_sorted.iloc[0]['time']]

        # Step 3: Find the row with the lowest 'predicted' value
        lowest_predicted_row = most_recent_records.loc[most_recent_records['predicted'].idxmin()]

        # Step 4: Print out the 'train_id' and 'predicted' value of the lowest predicted row
        train_id = lowest_predicted_row['train_id']
        predicted_value = lowest_predicted_row['predicted']

        print(f"Train ID: {train_id}, Predicted Value: {predicted_value}")
