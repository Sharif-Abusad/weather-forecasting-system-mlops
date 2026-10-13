"""
Model Pusher Module

Pushes accepted models to the deployment directory:
1. Checks model acceptance from evaluation artifact.
2. Copies LSTM model, XGBoost model, scaler and metadata from saved_models
   to deployment/ — the directory the FastAPI app reads from.
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
            
            # Create empty dir so DVC output exists
            create_directories([self.config.pushed_model_dir])

            if not self.model_evaluation_artifact.is_model_accepted:
                logger.warning(
                    f"Models rejected — skipping push. "
                    f"Reason: {self.model_evaluation_artifact.message}"
                )
                return ModelPusherArtifact(
                    pushed_model_dir = self.config.pushed_model_dir,
                    is_pushed       = False,
                    message         = "Models not pushed — evaluation thresholds not met.",
                )

            # Files to push: (source, destination)
            from src.constants import (
                SAVED_MODEL_DIR, 
                LSTM_MODEL_FILE, 
                XGB_MODEL_FILE, 
                LSTM_META_FILE, 
                XGB_META_FILE, 
                SCALER_FILE_NAME
            )
            files_to_push = [
                (os.path.join(SAVED_MODEL_DIR, LSTM_MODEL_FILE),  self.config.lstm_model_path),
                (os.path.join(SAVED_MODEL_DIR, XGB_MODEL_FILE),   self.config.xgb_model_path),
                (os.path.join(SAVED_MODEL_DIR, SCALER_FILE_NAME), self.config.scaler_path),
                (os.path.join(SAVED_MODEL_DIR, LSTM_META_FILE),   self.config.lstm_meta_path),
                (os.path.join(SAVED_MODEL_DIR, XGB_META_FILE),    self.config.xgb_meta_path),
            ]
            for src, dst in files_to_push:
                if os.path.exists(src):
                    self._copy_file(src, dst)
                else:
                    logger.warning(f"Source not found, skipping: {src}")

            artifact = ModelPusherArtifact(
                pushed_model_dir = self.config.pushed_model_dir,
                is_pushed       = True,
                message         = f"Models pushed -> {self.config.pushed_model_dir}",
            )

            logger.info("=" * 60)
            logger.info(f"Model Pusher Artifact: {artifact}")
            logger.info("=" * 60)
            return artifact

        except Exception as e:
            raise WeatherException(e, sys)

 
if __name__ == "__main__":
    from src.entity.config_entity import ModelPusherConfig, ModelEvaluationConfig
    from src.entity.artifact_entity import ModelEvaluationArtifact

    # Load eval report from reports/
    import json
    with open("reports/eval_report.json") as f:
        report = json.load(f)

    evaluation_artifact = ModelEvaluationArtifact(
        is_model_accepted = report["is_model_accepted"],
        lstm_mae          = report["new_model"]["lstm_mae"],
        xgb_auc           = report["new_model"]["xgb_auc"],
        report_file_path  = ModelEvaluationConfig().report_file_path,
        message           = report["acceptance_reason"],
    )

    pusher   = ModelPusher(ModelPusherConfig(), evaluation_artifact)
    artifact = pusher.initiate_model_pusher()