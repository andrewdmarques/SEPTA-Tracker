
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import re
from datetime import datetime
import pickle
import numpy as np

############################################################
# Input variables
############################################################

poi_heading = "E"
# Change to the correcting working directory
dir_working = '/home/andrewdmarques/Desktop/Train-Tracker'
file_in = 'toi.csv'
file_model = 'model.pkl'
model_train = True
model_predict = False

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
    toi4 = toi3.copy()
    toi5 = toi4[(toi4['train_heading_average'] == poi_heading)]



print('filtered to have just the trains of interest')

######################

if model_train: # If the linear regression is to be trained by the data.
    print('Beginning Training')

    # Prepare the training data further.
    # Filter to have just the cleanest trains -- removing those trains that take longer than expected.
    # Load the dataset
    toi6 = pd.read_csv('toi5.csv')

    # Identify the train_ids that have any row with time_to_poi greater than 30 or greater than 8
    train_ids_to_remove = toi6[(toi6['time_to_poi'] > 30) | (toi6['poi_dist'] > 8)]['train_id'].unique()

    # Filter the dataset to remove rows with those train_ids
    filtered_toi6 = toi6[~toi6['train_id'].isin(train_ids_to_remove)]
    
    # Filter to remove train groups that have do not really make it to the station.
    filtered_toi6 = filtered_toi6.groupby('train_id').filter(lambda x: x['poi_dist'].min() <= 0.25)

    # If you want to save it to a new CSV file
    filtered_toi6.to_csv('toi6.csv', index=False)
    

    ######################
    # Train the model and save it.
    # Load the dataset
    toi7 = pd.read_csv('toi6.csv')

    # Ensure the correct column names
    X = toi7[['poi_dist']]  # independent variable
    y = toi7['time_to_poi']     # dependent variable

    # Initialize and fit the linear regression model
    model = LinearRegression()
    model.fit(X, y)
    
    # Create a DataFrame with the predictions and actual values
    predictions = model.predict(X)
    results_df = pd.DataFrame({
        'prediction': predictions,
        'actual_value': y
    })

    # Save the DataFrame to a CSV file
    results_df.to_csv('prediction.csv', index=False)

    print('Completed training model')
    # Save the trained model using pickle
    with open(file_model, 'wb') as model_file:
        pickle.dump(model, model_file)
    
    # Assess how the model performed.
    # Predict the values using the trained model
    y_pred = model.predict(X)

    # Convert X to a 1D array for plotting
    X_plot = X.values.flatten()

    # Create a scatter plot of the actual data
    plt.scatter(X_plot, y, color='blue', label='Actual Data')

    # Plot the regression line
    plt.plot(X_plot, y_pred, color='red', label='Regression Line')

    # Add labels and title
    plt.xlabel('POI Distance')
    plt.ylabel('Time to POI')
    plt.title('Linear Regression: Time to POI vs POI Distance')

    # Add a legend
    plt.legend()

    # Show the plot
    plt.show()
    
    
    
        
        
    # Convert to DataFrame for easier filtering and manipulation
    results_df = pd.DataFrame({'actual': y, 'predicted': y_pred})

    # Filter the data to include only rows where actual time_to_poi is between 12 and 17
    filtered_df = results_df[(results_df['actual'] >= 12) & (results_df['actual'] <= 17)]

    # Calculate the difference between predicted and actual values (errors)
    filtered_df['error'] = filtered_df['predicted'] - filtered_df['actual']

    # Calculate the mean and standard deviation of the errors
    mean_error = np.mean(filtered_df['error'])
    std_error = np.std(filtered_df['error'])
    two_std = 2 * std_error  # Two standard deviations

    # Create a histogram of the errors
    plt.figure(figsize=(10, 6))
    plt.hist(filtered_df['error'], bins=20, color='skyblue', edgecolor='black')

    # Add titles and labels, including two standard deviation values in the title
    plt.title(f'Histogram of Prediction Errors (Actual time_to_poi between 12 and 17)\n'
              f'Mean Error: {mean_error:.2f}, 2 Std Dev: ±{two_std:.2f}')
    plt.xlabel('Prediction Error (Predicted - Actual)')
    plt.ylabel('Frequency')

    # Show the plot
    plt.show()




            
if model_predict == True:
    # To load the model for future predictions
    with open(file_model, 'rb') as model_file:
        loaded_model = pickle.load(model_file)
    
    
        


