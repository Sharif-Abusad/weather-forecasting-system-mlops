from dataclasses import dataclass
 
 
@dataclass
class DataIngestionArtifact:
    raw_file_path: str
    is_ingested: bool
    message: str