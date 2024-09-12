import pandas as pd
import os

# Define the folder containing the CSV files
folder_path = './test'

# Initialize variables to calculate overall average inference time
total_inference_time = 0
total_files = 0

# Loop through all files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        # Construct the full file path
        file_path = os.path.join(folder_path, filename)
        
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path)
        df = df.fillna(0)

        # Calculate the mean accuracy for this file
        mean_accuracy = df['r2_zero_final_predict'].mean()
        
        # Calculate the mean accuracy for direct prediction without latent space optimization
        mean_direct_accuracy = df['r2_zero_direct_predict'].mean()
        
        # Calculate the mean model complexity for this file
        mean_complexity = df['_complexity_final_predict'].mean()
        
        # Calculate the average inference time for this file
        average_inference_time = df['time'].mean()
        
        # Update the total inference time and file count
        total_inference_time += average_inference_time
        total_files += 1
        
        # Print the mean accuracy, direct prediction accuracy, model complexity, and average inference time for this file
        print(f"File: {filename}")
        print(f"  Mean accuracy: {mean_accuracy}")
        print(f"  Mean direct prediction accuracy: {mean_direct_accuracy}")
        print(f"  Mean model complexity: {mean_complexity}")
        print(f"  Average inference time: {average_inference_time}\n")

# Calculate the overall average inference time across all files
overall_average_inference_time = total_inference_time / total_files

print(f"Overall average inference time across all files: {overall_average_inference_time}")