import shutil
import glob
import os
import argparse
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ******************
RATIO = 0.1
# ******************

def move_files(file_list, source_dir, dest_dir, desc="Moving files"):
    """Move files instead of copying (much faster if on same filesystem)"""
    for filename in tqdm(file_list, desc=desc):
        src = os.path.join(source_dir, filename)
        dst = os.path.join(dest_dir, filename)
        shutil.move(src, dst)

if __name__ == "__main__":
    source = "/home/lar/Riccardo/dlo_pyel/dataset_20250901_132645"
    destination = "/home/lar/Riccardo/dlo_diffusion/DATA_500k" 
    
    train_path = os.path.join(destination, "train")
    val_path = os.path.join(destination, "val")
    
    
    # Create directories
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(val_path, exist_ok=True)
    
    # Get all pickle files
    files_list = glob.glob(os.path.join(source, "*.pkl"))
    files_names = [f.split("/")[-1] for f in files_list]
    
    # Split into train and test
    train, test = train_test_split(files_names, test_size=RATIO, random_state=42)
    
    print(f"Train size: {len(train)}, Test size: {len(test)}")
    print("Moving files (fastest option)...")
    print("WARNING: Source directory will be empty after this operation!")
    
    # Confirm before proceeding
    response = input("Continue? (y/N): ").strip().lower()
    if response not in ['y', 'yes']:
        print("Operation cancelled.")
        exit(0)
    
    # Move files (this empties the source directory)
    move_files(train, source, train_path, "Moving training samples")
    move_files(test, source, val_path, "Moving validation samples")
    
    print("Dataset splitting completed!")
    print(f"Original files moved from: {source}")
    print(f"Training files in: {train_path}")
    print(f"Validation files in: {val_path}")
    print(f"\nTo merge back, use the reverse script with these paths.")