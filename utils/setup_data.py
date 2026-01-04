import zipfile
import os

def setup_data():
    # Get the path to the 'project' root (2 levels up from utils)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    # Define paths relative to the project root
    zip_file_path = os.path.join(project_root, 'data', 'SPair-71k.zip') 
    extract_dir = os.path.join(project_root, 'data', 'SPair-71k_extracted')

    # Create extraction directory
    os.makedirs(extract_dir, exist_ok=True)

    # Check if we need to extract
    if not os.path.exists(zip_file_path):
        print(f"Error: Zip file not found at {zip_file_path}")
        return None

    # Check if already extracted (simple check)
    if os.path.exists(os.path.join(extract_dir, 'SPair-71k')): 
        print(f"Data already extracted in {extract_dir}")
        return extract_dir

    print(f"Extracting {zip_file_path}...")
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print(f"Extraction complete!")
    
    return extract_dir

if __name__ == "__main__":
    setup_data()