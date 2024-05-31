import pandas as pd
import numpy as np
import os

def process_mafaulda_data(dataset_dir, n_rows=250000, n_cols=8):
    # Initialize lists to store data arrays, labels, and class IDs
    data_arrays = []
    labels = []
    class_ids = []

    # Class ID counter
    class_id_counter = 0

    # Function to process and load a CSV file
    def process_csv_file(file_path):
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Ensure the data is of the correct shape (250000 x 8)
        if df.shape[0] > n_rows:
            df = df.iloc[:n_rows, :]
        elif df.shape[0] < n_rows:
            df = df.reindex(np.arange(n_rows)).fillna(0)
        
        if df.shape[1] > n_cols:
            df = df.iloc[:, :n_cols]
        elif df.shape[1] < n_cols:
            df = df.reindex(columns=np.arange(n_cols)).fillna(0)
        
        return df.values

    # Iterate over each folder in the dataset directory
    for root, dirs, files in os.walk(dataset_dir):
        for dir_name in dirs:
            folder_path = os.path.join(root, dir_name)
            
            # Iterate over each CSV file in the folder
            for file_name in os.listdir(folder_path):
                if file_name.endswith('.csv'):
                    file_path = os.path.join(folder_path, file_name)
                    
                    # Process and load the CSV file
                    data_array = process_csv_file(file_path)
                    
                    # Append data array, label, and class ID to respective lists
                    data_arrays.append(data_array)
                    labels.append(dir_name)
                    class_ids.append(class_id_counter)
            
            # Increment class ID counter for the next fault type
            class_id_counter += 1

    # Convert lists to numpy arrays
    data_arrays = np.array(data_arrays)
    labels = np.array(labels)
    class_ids = np.array(class_ids)

    # Save the data arrays and labels to files
    np.save('data_arrays.npy', data_arrays)
    np.save('labels.npy', labels)
    np.save('class_ids.npy', class_ids)

    return data_arrays, labels, class_ids