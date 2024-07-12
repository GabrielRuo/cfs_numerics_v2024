import os
import csv

def extract_MF(folder_name):
    """Extract M and F from the folder name."""
    try:
        parts = folder_name.split('-')
        M = int(parts[0].split('_')[1])  # Extract M
        F = int(parts[1].split('_')[1])  # Extract F
        return M, F
    except Exception as e:
        print(f"Error extracting M and F from folder name {folder_name}: {e}")
        raise

def process_slurm_err(file_path):
    """Process the slurm.err file and return the runtime or the second to last line."""
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            if len(lines) < 2:
                print(f"File {file_path} does not have enough lines.")
                return None
            
            # Check the last ten lines for "The script took"
            last_ten_lines = lines[-30:]
            for line in last_ten_lines:
                if "The script took" in line:
                    runtime = line.split("The script took")[-1].strip().replace(" to run.", "")
                    return runtime
            
            # If not found, return the second to last line
            return lines[-2].strip()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None

def main(parent_folder, output_csv):
    results = []
    print(f"Scanning parent folder: {parent_folder}")
    for folder_name in os.listdir(parent_folder):
        folder_path = os.path.join(parent_folder, folder_name)
        if os.path.isdir(folder_path) and folder_name.startswith('m_') and '-f_' in folder_name:
            print(f"Processing folder: {folder_name}")
            try:
                M, F = extract_MF(folder_name)
                slurm_err_path = os.path.join(folder_path, 'slurm.err')
                if os.path.exists(slurm_err_path):
                    result = process_slurm_err(slurm_err_path)
                    if result:
                        results.append((M, F, result))
                        print(f"Added result: ({M}, {F}, {result})")
                    else:
                        print(f"No valid result in file: {slurm_err_path}")
                else:
                    print(f"File not found: {slurm_err_path}")
            except Exception as e:
                print(f"Error processing folder {folder_name}: {e}")

    # Sort results by M and F
    results.sort(key=lambda x: (x[0], x[1]))

    print(f"Writing results to CSV: {output_csv}")
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['M', 'F', 'Result'])
        writer.writerows(results)
    print(f"Results written to {output_csv}")

if __name__ == "__main__":
    parent_folder = os.path.dirname(os.path.abspath(__file__))  # The folder the script is running in
    output_csv = os.path.join(parent_folder, "output3.csv")
    main(parent_folder, output_csv)
