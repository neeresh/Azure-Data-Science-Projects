
# Import libraries
import os
import argparse
import pandas as pd
from azureml.core import Run
from sklearn.preprocessing import MinMaxScaler

# Get Parameters
parser = argparse.ArgumentParser()
parser.add_argument('--input-data', type = str, dest = 'raw_dataset_id', help = 'Raw Dataset')
parser.add_argument('--prepped-data', type = str, dest = 'prepped_data', help = 'Folder for results')
args = parser.parse_args()

save_folder = args.prepped_data

# Get the Experiment Run Context
run = Run.get_context()

# Load the data
diabetes = run.input_datasets['raw_data'].to_pandas_dataframe()

# Log raw row count
row_count = (len(diabetes))
run.log('raw_rows', row_count)

# Remove missing data from the dataset
diabetes = diabetes.dropna()

# Applying MinMaxScaler
scaler = MinMaxScaler()
num_cols = ['Pregnancies','PlasmaGlucose','DiastolicBloodPressure','TricepsThickness','SerumInsulin','BMI','DiabetesPedigree']
diabetes[num_cols] = scaler.fit_transform(diabetes[num_cols])

# Log processed rows
row_count = (len(diabetes))
run.log('processed_rows', row_count)

# Save the prepped data
print("Saving Data...")
os.makedirs(save_folder, exist_ok=True)
save_path = os.path.join(save_folder,'data.csv')
diabetes.to_csv(save_path, index=False, header=True)

# End the run
run.complete()
