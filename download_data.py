import kagglehub
import os
import shutil

def download_dataset():
    print("Downloading FaceForensics++ (C23) dataset...")
    try:
        # Download latest version
        path = kagglehub.dataset_download("xdxd003/ff-c23")
        print(f"Dataset downloaded to: {path}")
        
        # Define target directory
        target_dir = os.path.join(os.getcwd(), "dataset")
        
        # Check if we need to move/copy it or if we can use it in place
        # For simplicity, let's just print the path for now, as moving huge datasets can be slow
        # But for the training script, we'll need to know where it is.
        
        print("\nIMPORTANT: Please note the path above.")
        print("You can either move the files to a 'dataset' folder in this project,")
        print("or update the DATA_DIR in train.py to point to the downloaded path.")
        
        # Create a marker file so we know it's done
        with open("dataset_info.txt", "w") as f:
            f.write(path)
            
    except Exception as e:
        print(f"Error downloading dataset: {e}")

if __name__ == "__main__":
    download_dataset()
