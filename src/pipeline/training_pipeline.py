"""
Training Pipeline

Runs all pipeline stages in sequence:
1. Data Ingestion
2. Data Transformation
3. Data Validation
4. Model Trainer
5. Model Evaluation
6. Model Pusher
"""

import sys
from src.logger import logger
from src.exception import WeatherException

from src.entity.config_entity import (
    DataIngestionConfig,
    DataTransformationConfig,
    DataValidationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig,
    ModelPusherConfig,
)
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.data_validation import DataValidation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation
from src.components.model_pusher import ModelPusher


class TrainingPipeline:

    def __init__(self):
        self.data_ingestion_config      = DataIngestionConfig()
        self.data_transformation_config = DataTransformationConfig()
        self.data_validation_config     = DataValidationConfig()
        self.model_trainer_config       = ModelTrainerConfig()
        self.model_evaluation_config    = ModelEvaluationConfig()
        self.model_pusher_config        = ModelPusherConfig()

    # ── Stage 1 ──────────────────────────────────────────────
    def start_data_ingestion(self):
        try:
            logger.info(">>> Stage 1: Data Ingestion <<<")
            ingestion = DataIngestion(self.data_ingestion_config)
            artifact  = ingestion.initiate_data_ingestion()
            logger.info(f">>> Stage 1 complete: {artifact} <<<")
            return artifact
        except Exception as e:
            raise WeatherException(e, sys)

    # ── Stage 2 ──────────────────────────────────────────────
    def start_data_transformation(self, ingestion_artifact):
        try:
            logger.info(">>> Stage 2: Data Transformation <<<")
            transformation = DataTransformation(
                self.data_transformation_config,
                ingestion_artifact,
            )
            artifact = transformation.initiate_data_transformation()
            logger.info(f">>> Stage 2 complete: {artifact} <<<")
            return artifact
        except Exception as e:
            raise WeatherException(e, sys)

    # ── Stage 3 ──────────────────────────────────────────────
    def start_data_validation(self, transformation_artifact):
        try:
            logger.info(">>> Stage 3: Data Validation <<<")
            validation = DataValidation(
                self.data_validation_config,
                transformation_artifact,
            )
            artifact = validation.initiate_data_validation()
            logger.info(f">>> Stage 3 complete: {artifact} <<<")
            return artifact
        except Exception as e:
            raise WeatherException(e, sys)

    # ── Stage 4 ──────────────────────────────────────────────
    def start_model_trainer(self, validation_artifact):
        try:
            logger.info(">>> Stage 4: Model Trainer <<<")
            trainer  = ModelTrainer(
                self.model_trainer_config,
                validation_artifact,
            )
            artifact = trainer.initiate_model_trainer()
            logger.info(f">>> Stage 4 complete: {artifact} <<<")
            return artifact
        except Exception as e:
            raise WeatherException(e, sys)

    # ── Stage 5 ──────────────────────────────────────────────
    def start_model_evaluation(self, trainer_artifact):
        try:
            logger.info(">>> Stage 5: Model Evaluation <<<")
            evaluation = ModelEvaluation(
                self.model_evaluation_config,
                trainer_artifact,
            )
            artifact = evaluation.initiate_model_evaluation()
            logger.info(f">>> Stage 5 complete: {artifact} <<<")
            return artifact
        except Exception as e:
            raise WeatherException(e, sys)

    # ── Stage 6 ──────────────────────────────────────────────
    def start_model_pusher(self, evaluation_artifact):
        try:
            logger.info(">>> Stage 6: Model Pusher <<<")
            pusher   = ModelPusher(
                self.model_pusher_config,
                evaluation_artifact,
            )
            artifact = pusher.initiate_model_pusher()
            logger.info(f">>> Stage 6 complete: {artifact} <<<")
            return artifact
        except Exception as e:
            raise WeatherException(e, sys)

    # ── Run full pipeline ─────────────────────────────────────
    def run(self):
        try:
            logger.info("=" * 60)
            logger.info("TRAINING PIPELINE STARTED")
            logger.info("=" * 60)

            ingestion_artifact      = self.start_data_ingestion()
            transformation_artifact = self.start_data_transformation(ingestion_artifact)
            validation_artifact     = self.start_data_validation(transformation_artifact)
            trainer_artifact        = self.start_model_trainer(validation_artifact)
            evaluation_artifact     = self.start_model_evaluation(trainer_artifact)
            pusher_artifact         = self.start_model_pusher(evaluation_artifact)

            logger.info("=" * 60)
            logger.info("TRAINING PIPELINE COMPLETED")
            logger.info(f"Models pushed : {pusher_artifact.is_pushed}")
            logger.info(f"Saved to      : {pusher_artifact.saved_model_dir}")
            logger.info("=" * 60)

            return pusher_artifact

        except Exception as e:
            raise WeatherException(e, sys)


if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.run()