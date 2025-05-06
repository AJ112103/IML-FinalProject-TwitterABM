import os
import sys
import logging
import subprocess
from pathlib import Path

os.makedirs(os.path.join(os.path.dirname(__file__), '..', '..', 'logs'), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(os.path.dirname(__file__), '..', '..', 'logs', 'download.log'), mode='w')
    ]
)
logger = logging.getLogger('download')

DATASET_NAME = "mulengakawimbe89/in-depth-twitter-retweet-analysis-dataset"

def check_kaggle_credentials():
    cred_path = Path.home() / '.kaggle' / 'kaggle.json'
    if not cred_path.exists():
        logger.error(f"Kaggle credentials not found at {cred_path}")
        logger.info("Please obtain API credentials from https://www.kaggle.com/account")
        logger.info("and place kaggle.json in the ~/.kaggle/ directory")
        logger.info("Then run: chmod 600 ~/.kaggle/kaggle.json")
        sys.exit(1)

    if oct(cred_path.stat().st_mode)[-3:] != '600':
        logger.warning(f"Kaggle credentials file has incorrect permissions: {oct(cred_path.stat().st_mode)[-3:]}")
        logger.warning("For security, permissions should be 600. Fixing...")
        try:
            cred_path.chmod(0o600)
            logger.info("Permissions fixed.")
        except Exception as e:
            logger.error(f"Failed to fix permissions: {e}")
            logger.info("Please run: chmod 600 ~/.kaggle/kaggle.json")
            sys.exit(1)
    
    return True

def download_dataset(output_dir="data/raw"):
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Downloading dataset {DATASET_NAME} to {output_dir}...")
    try:
        subprocess.run(
            ["kaggle", "datasets", "download", DATASET_NAME, "-p", output_dir, "--unzip"],
            check=True
        )
        logger.info("Download completed successfully")
        try:
            csv_path = os.path.join(output_dir, "In-Depth Twitter Retweet Analysis Dataset.csv")
            new_path = os.path.join(output_dir, "retweet_analysis.csv")
            if os.path.exists(csv_path):
                os.rename(csv_path, new_path)
                logger.info(f"Renamed dataset file to {new_path}")
        except Exception as e:
            logger.warning(f"Could not rename dataset file: {e}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to download dataset: {e}")
        sys.exit(1)

def main():
    logger.info("Starting dataset download process")

    if check_kaggle_credentials():
        download_dataset()
        logger.info("Dataset download process completed")

if __name__ == "__main__":
    main() 