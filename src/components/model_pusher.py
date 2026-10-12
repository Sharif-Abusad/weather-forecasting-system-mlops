"""
Model Pusher Module

Pushes accepted models to the saved_models directory:
1. Checks model acceptance from evaluation artifact.
2. Copies LSTM model, XGBoost model, scaler and metadata
   to saved_models/ — the directory the FastAPI app reads from.
3. Versions the saved models with a timestamp.
4. Returns a ModelPusherArtifact for downstream pipeline stages.
"""

import os
import sys
import shutil

from src.logger import logger
from src.exception import WeatherException
from src.entity.config_entity import ModelPusherConfig
from src.entity.artifact_entity import ModelPusherArtifact, ModelEvaluationArtifact
from src.utils.main_utils import create_directories


class ModelPusher:

    def __init__(
        self,
        config: ModelPusherConfig,
        model_evaluation_artifact: ModelEvaluationArtifact,
    ):
        self.config                    = config
        self.model_evaluation_artifact = model_evaluation_artifact

    def _copy_file(self, src: str, dst: str) -> None:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        logger.info(f"Copied -> {dst}")

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        try:
            logger.info("=" * 60)
            logger.info("Starting Model Pusher")
            logger.info("=" * 60)

            if not self.model_evaluation_artifact.is_model_accepted:
                logger.warning(
                    f"Models rejected — skipping push. "
                    f"Reason: {self.model_evaluation_artifact.message}"
                )
                return ModelPusherArtifact(
                    saved_model_dir = self.config.saved_model_dir,
                    is_pushed       = False,
                    message         = "Models not pushed — evaluation thresholds not met.",
                )

            create_directories([self.config.saved_model_dir])

            # Files to push: (source, destination)
            files_to_push = [
                (
                    self.model_evaluation_artifact.__dict__.get(
                        "lstm_model_path",
                        os.path.join(
                            os.path.dirname(self.model_evaluation_artifact.report_file_path)
                            .replace("model_evaluation", "model_trainer"),
                            "lstm_temp_model.keras",
                        ),
                    ),
                    self.config.lstm_model_path,
                ),
                (
                    os.path.join(
                        os.path.dirname(self.model_evaluation_artifact.report_file_path)
                        .replace("model_evaluation", "model_trainer"),
                        "rain_model.json",
                    ),
                    self.config.xgb_model_path,
                ),
                (
                    os.path.join(
                        os.path.dirname(self.model_evaluation_artifact.report_file_path)
                        .replace("model_evaluation", "model_trainer"),
                        "lstm_temp_scaler.pkl",
                    ),
                    self.config.scaler_path,
                ),
                (
                    os.path.join(
                        os.path.dirname(self.model_evaluation_artifact.report_file_path)
                        .replace("model_evaluation", "model_trainer"),
                        "lstm_model_meta.json",
                    ),
                    self.config.lstm_meta_path,
                ),
                (
                    os.path.join(
                        os.path.dirname(self.model_evaluation_artifact.report_file_path)
                        .replace("model_evaluation", "model_trainer"),
                        "xgb_model_meta.json",
                    ),
                    self.config.xgb_meta_path,
                ),
            ]

            for src, dst in files_to_push:
                if os.path.exists(src):
                    self._copy_file(src, dst)
                else:
                    logger.warning(f"Source not found, skipping: {src}")

            artifact = ModelPusherArtifact(
                saved_model_dir = self.config.saved_model_dir,
                is_pushed       = True,
                message         = f"Models pushed -> {self.config.saved_model_dir}",
            )

            logger.info("=" * 60)
            logger.info(f"Model Pusher Artifact: {artifact}")
            logger.info("=" * 60)
            return artifact

        except Exception as e:
            raise WeatherException(e, sys)

 
if __name__ == "__main__":
    from src.entity.config_entity import (
        DataIngestionConfig, DataTransformationConfig,
        DataValidationConfig, ModelTrainerConfig,
        ModelEvaluationConfig, ModelPusherConfig,
    )
    from src.components.data_ingestion import DataIngestion
    from src.components.data_transformation import DataTransformation
    from src.components.data_validation import DataValidation
    from src.components.model_trainer import ModelTrainer
    from src.components.model_evaluation import ModelEvaluation

    ingestion_artifact      = DataIngestion(DataIngestionConfig()).initiate_data_ingestion()
    transformation_artifact = DataTransformation(DataTransformationConfig(), ingestion_artifact).initiate_data_transformation()
    validation_artifact     = DataValidation(DataValidationConfig(), transformation_artifact).initiate_data_validation()
    trainer_artifact        = ModelTrainer(ModelTrainerConfig(), validation_artifact).initiate_model_trainer()
    evaluation_artifact     = ModelEvaluation(ModelEvaluationConfig(), trainer_artifact).initiate_model_evaluation()

    pusher   = ModelPusher(ModelPusherConfig(), evaluation_artifact)
    artifact = pusher.initiate_model_pusher()