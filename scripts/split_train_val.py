import shutil
import glob
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ******************
RATIO = 0.1
PERCENTAGE_OF_DATASET = 1
# ******************

def copy_files(file_list, source_dir, dest_dir, desc="Copying files"):
    """Copy files instead of moving"""
    for filename in tqdm(file_list, desc=desc):
        src = os.path.join(source_dir, filename)
        dst = os.path.join(dest_dir, filename)
        shutil.copy2(src, dst)  # copy2 preserves metadata


if __name__ == "__main__":
    source = "/home/mengo/research/dlo_diffusion/train_episodes"
    destination = "/home/mengo/research/dlo_diffusion/train_episodes_splitted" 
    
    train_path = os.path.join(destination, "train")
    val_path = os.path.join(destination, "val")
    
    # Create directories
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(val_path, exist_ok=True)
    
    # Get all pickle files
    files_list = glob.glob(os.path.join(source, "*.pkl"))
    files_names = [os.path.basename(f) for f in files_list]
    files_names = files_names[:int(len(files_names)*PERCENTAGE_OF_DATASET)]
    
    # Split into train and test
    train, test = train_test_split(files_names, test_size=RATIO, random_state=42)
    
    print(f"Train size: {len(train)}, Test size: {len(test)}")
    print("Copying files (source directory will remain intact)...")
    
    # Confirm before proceeding
    response = input("Continue? (y/N): ").strip().lower()
    if response not in ['y', 'yes']:
        print("Operation cancelled.")
        exit(0)
    
    # Copy files
    copy_files(train, source, train_path, "Copying training samples")
    copy_files(test, source, val_path, "Copying validation samples")
    
    print("Dataset splitting completed!")
    print(f"Original files kept in: {source}")
    print(f"Training files copied to: {train_path}")
    print(f"Validation files copied to: {val_path}")
