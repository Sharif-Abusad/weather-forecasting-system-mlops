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


@dataclass
class DataValidationArtifact:
    validation_status: bool
    report_file_path: str
    message: str


@dataclass
class ModelTrainerArtifact:
    lstm_model_path: str
    xgb_model_path: str
    lstm_meta_path: str
    xgb_meta_path: str
    scaler_path: str
    lstm_mae: float
    xgb_auc: float
    is_trained: bool
    message: str


@dataclass
class ModelEvaluationArtifact:
    is_model_accepted: bool
    lstm_mae: float
    xgb_auc: float
    report_file_path: str
    message: str


@dataclass
class ModelPusherArtifact:
    saved_model_dir: str
    is_pushed: bool
    message: str