
import shutil
import time
import logging
import pandas as pd
from pathlib import Path
from TrackAnalyzer import TrackAnalyzer
from utils import get_columns

last_backup_time = time.time()
backup_interval_time = 15 * 60 

data_file = "data/track_features.csv"
backup_file = "data/track_features_backup.csv"
log_file = "data/processing_log.txt"
path = r"D:\Music\DJ\Electronic"
root_directory = Path(path)

logging.basicConfig(
    filename="error.log",
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Write columns if file doesn't exist
columns = get_columns()
if not Path(data_file).exists():
    pd.DataFrame(columns=columns).to_csv(data_file, index=False)

# Process files
for mp3_file in root_directory.rglob("*.mp3"):
    try:
        start_time = time.time()

        full_path = mp3_file.as_posix()
        print(f"Processing file: {mp3_file.name} ....")

        with open(log_file, "a", encoding="utf-8") as log:
            log.write(f"Processing: {full_path}\n")
        
        analyzer = TrackAnalyzer(full_path)
        analyzer.extract_features()
        features = analyzer.get_features()

        # Append to CSV using Pandas
        row = pd.DataFrame([features], columns=columns)
        row.to_csv(data_file, mode="a", header=False, index=False)
        
        elapsed_time = time.time() - start_time
        print(f"Processed {mp3_file.name} in {elapsed_time:.2f} seconds")

        current_time = time.time()
        if (current_time - last_backup_time) >= backup_interval_time:
            shutil.copy2(data_file, backup_file)  # Create a backup
            print(f"Backup created at {time.strftime('%Y-%m-%d %H:%M:%S')}")
            last_backup_time = current_time
    
    except Exception as e:
        logging.error(f"Error processing file {mp3_file}: {e}")
        continue