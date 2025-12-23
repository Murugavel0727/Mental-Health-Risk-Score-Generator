import kagglehub
import shutil
import os

def download_data():
    print("Downloading dataset...")
    try:
        # Download latest version
        path = kagglehub.dataset_download("sahideseker/mental-health-risk-prediction-dataset")
        print("Path to dataset files:", path)
        
        # Move to local ml/data folder for easier access
        target_dir = os.path.join(os.path.dirname(__file__), "data")
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
            
        # Copy files
        for item in os.listdir(path):
            s = os.path.join(path, item)
            d = os.path.join(target_dir, item)
            if os.path.isfile(s):
                shutil.copy2(s, d)
                print(f"Copied {item} to {target_dir}")
                
    except Exception as e:
        print(f"Error downloading data: {e}")

if __name__ == "__main__":
    download_data()
