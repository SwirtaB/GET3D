import numpy as np
import pandas as pd
import os

# Folder containing the .npy files
DATA_FOLDER = '/home/michal/Desktop/GET3D/save_inference_results/shapenet_car/inference/mesh_pred'

# Existing CSV file to append the data
CSV_FILE = '/home/michal/Desktop/GET3D/data.csv'

# Create an empty list to hold the dictionaries
data_list = []

# Iterate over the files in the data folder
for file_name in os.listdir(DATA_FOLDER):
    if file_name.endswith('.npy'):
        file_path = os.path.join(DATA_FOLDER, file_name)
        # Load the .npy file using numpy
        data = np.load(file_path)
        # Convert the loaded data to a dictionary with "object" and "class" columns
        data_dict = {'object': data[0][0], 'class': ''}
        # Append the dictionary to the list
        data_list.append(data_dict)

# Create the DataFrame from the list of dictionaries
df = pd.DataFrame(data_list, columns=['object', 'class'])

# Check if the CSV file exists
if os.path.isfile(CSV_FILE):
    # Append only the "object" column to the existing CSV file
    df[['object']].to_csv(CSV_FILE, mode='a', header=False, index=False)
else:
    # Write the entire DataFrame to the CSV file
    df.to_csv(CSV_FILE, index=False)
