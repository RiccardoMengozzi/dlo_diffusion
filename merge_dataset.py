import shutil
import glob
import os
from tqdm import tqdm

def move_files_back(source_dir, dest_dir, desc="Moving files"):
    """Move all files from source_dir back to dest_dir"""
    if not os.path.exists(source_dir):
        print(f"Warning: {source_dir} does not exist. Skipping.")
        return 0
    
    files_list = glob.glob(os.path.join(source_dir, "*.pkl"))
    
    if not files_list:
        print(f"No .pkl files found in {source_dir}")
        return 0
    
    for file_path in tqdm(files_list, desc=desc):
        filename = os.path.basename(file_path)
        dest_file = os.path.join(dest_dir, filename)
        
        # Check if file already exists in destination
        if os.path.exists(dest_file):
            print(f"Warning: {filename} already exists in destination. Skipping.")
            continue
            
        shutil.move(file_path, dest_file)
    
    return len(files_list)

if __name__ == "__main__":
    # Configuration - UPDATE THESE PATHS TO MATCH YOUR SETUP
    original_source = "/home/lar/Riccardo/dlo_pyel/dataset_20250901_132645"
    dataset_base_path = "/home/lar/Riccardo/dlo_diffusion"
    dataset_name = "DATA_500k"
    # Construct paths
    train_path = os.path.join(dataset_base_path, f"{dataset_name}", "train")
    val_path = os.path.join(dataset_base_path, f"{dataset_name}", "val")
    
    print("Dataset Merge Script")
    print("===================")
    print(f"Train folder: {train_path}")
    print(f"Validation folder: {val_path}")
    print(f"Target (original) folder: {original_source}")
    
    # Create original directory if it doesn't exist
    os.makedirs(original_source, exist_ok=True)
    
    # Check if directories exist
    if not os.path.exists(train_path) and not os.path.exists(val_path):
        print("Error: Neither train nor validation directories exist!")
        exit(1)
    
    # Show what we're about to do
    train_files = len(glob.glob(os.path.join(train_path, "*.pkl"))) if os.path.exists(train_path) else 0
    val_files = len(glob.glob(os.path.join(val_path, "*.pkl"))) if os.path.exists(val_path) else 0
    total_files = train_files + val_files
    
    print(f"\nFiles to merge:")
    print(f"  Training files: {train_files}")
    print(f"  Validation files: {val_files}")
    print(f"  Total files: {total_files}")
    
    if total_files == 0:
        print("No files to merge!")
        exit(0)
    
    # Confirm before proceeding
    print(f"\nThis will move all files back to: {original_source}")
    response = input("Continue? (y/N): ").strip().lower()
    if response not in ['y', 'yes']:
        print("Operation cancelled.")
        exit(0)
    
    # Move files back
    moved_train = move_files_back(train_path, original_source, "Moving training files back")
    moved_val = move_files_back(val_path, original_source, "Moving validation files back")
    
    total_moved = moved_train + moved_val
    
    print(f"\nMerge completed!")
    print(f"Files moved back: {total_moved}")
    print(f"Files now in: {original_source}")
    
    # Clean up empty directories (optional)
    cleanup = input("\nRemove empty train/val directories? (y/N): ").strip().lower()
    if cleanup in ['y', 'yes']:
        try:
            if os.path.exists(train_path) and not os.listdir(train_path):
                os.rmdir(train_path)
                print(f"Removed empty directory: {train_path}")
            if os.path.exists(val_path) and not os.listdir(val_path):
                os.rmdir(val_path)
                print(f"Removed empty directory: {val_path}")
        except Exception as e:
            print(f"Could not remove directories: {e}")
    
    print("Done!")