from dataclasses import dataclass
from src.constants import *

@dataclass
class DataIngestionConfig:
    artifact_dir: str = os.path.join(ARTIFACT_DIR, TIMESTAMP, DATA_INGESTION_DIR)
    raw_file_path: str = os.path.join(artifact_dir, DATA_INGESTION_RAW_DIR, RAW_FILE_NAME)
    archive_dir: str = os.path.join("data", "archives")   # ZIPs live here, git ignored
    raw_dir: str = os.path.join("data", "raw")            # extracted NetCDFs live here