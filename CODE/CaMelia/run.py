#Take GSE56879 as an example
import os
import subprocess

def run_commands_on_groups(directory, groups):
    """
    Run commands for three specified groups (2i, MII, Ser) using corresponding result files
    
    Parameters:
    directory: Root directory containing the three group subfolders
    groups: List of group names to process
    """
    for group_name in groups:
        group_folder = os.path.join(directory, group_name)
        processed_file = f"{group_name}.txt"
        file_path = os.path.join(group_folder, processed_file)        
        print(f"\nStarting processing group: {group_name}, using file: {processed_file}")
        
        # Define commands to run
        commands = [
            f"python CaMelia/Feature-extraction/get_local_Feature_for_train.py {group_folder} {processed_file} 10 0.8",
            f"python CaMelia/Feature-extraction/get_neighbor_Feature_for_train.py {group_folder} {processed_file} 10 0.8",
            f"python CaMelia/Feature-extraction/get_neighbor_Feature_for_imputation.py {group_folder} {processed_file} 10",
            f"python CaMelia/Feature-extraction/get_local_Feature_for_imputation.py {group_folder} {processed_file} 10",
            f"python CaMelia/Feature-extraction/unionfeature_for_train.py {group_folder} {processed_file} 10",
            f"python CaMelia/Feature-extraction/unionfeature_for_imputation.py {group_folder} {processed_file} 10",
            f"python CaMelia/Model-construction/model_TrainingandImputing.py {group_folder} {processed_file} CPU"
        ]
        
        for cmd in commands:
            print(f"Executing command: {cmd}")
            try:
                # Run command and capture output (compatible with Python 3.6)
                result = subprocess.run(
                    cmd, 
                    shell=True, 
                    check=True, 
                    stdout=subprocess.PIPE,  # Capture standard output
                    stderr=subprocess.PIPE,  # Capture error output
                    universal_newlines=True  # Alternative to text=True, works in Python 3.6
                )
                print(f"Command executed successfully: {cmd}")
            except subprocess.CalledProcessError as e:
                print(f"Command execution failed: {cmd}, error message: {e.stderr}")
        
        print(f"Group {group_name} processing completed\n{'='*50}")

# Example usage
if __name__ == "__main__":
    root_directory = 'CaMelia-master/GSE56879/'
    target_groups = ['2i']
    run_commands_on_groups(root_directory, target_groups)    
    print("All group processing commands have been executed")