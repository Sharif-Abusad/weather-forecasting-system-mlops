from dataclasses import dataclass
 
 
@dataclass
class DataIngestionArtifact:
    raw_file_path: str
    is_ingested: bool
    message: str


@dataclass
class DataTransformationArtifact:
    clean_file_path: str
    final_features_path: str
    is_transformed: bool
    message: str