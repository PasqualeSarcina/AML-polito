import os

def setup_data():
    """
    Returns the root directory for the extraction dataset folder (SPair-71k_extracted).
    Used by DINOv2 training script to locate the pair annotations, images, etc.
    """
    # Get the directory where this file (setup_data.py) is located (utils)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Get the project root (AML-polito)
    project_root = os.path.dirname(current_dir)
    
    # Path to the SPair-71k_extracted dataset directory
    data_root = os.path.join(project_root, 'dataset', 'SPair-71k_extracted')
    
    if os.path.exists(data_root):
        return data_root
        
    print(f"Warning: Dataset path {data_root} does not exist.")
    return None