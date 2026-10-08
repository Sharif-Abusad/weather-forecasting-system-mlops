"""
Data Validation Module

Validates the transformed weather dataset against schema.yaml:
1. Checks all required columns are present.
2. Validates value ranges for numerical columns.
3. Checks for missing values.
4. Validates rain_tomorrow is binary.
5. Checks temporal continuity (hourly).
6. Generates a validation report.
7. Returns a DataValidationArtifact for downstream pipeline stages.
"""

import os
import sys
import pandas as pd
from src.logger import logger
from src.exception import WeatherException
from src.entity.config_entity import DataValidationConfig
from src.entity.artifact_entity import DataValidationArtifact, DataTransformationArtifact
from src.utils.main_utils import read_yaml_file, save_json, create_directories


class DataValidation:

    def __init__(
        self,
        config: DataValidationConfig,
        data_transformation_artifact: DataTransformationArtifact,
    ):
        self.config = config
        self.data_transformation_artifact = data_transformation_artifact
        self.schema = read_yaml_file(self.config.schema_file_path)
        self.validation_errors = []

    # ── Load ─────────────────────────────────────────────────────
    def _load_data(self) -> pd.DataFrame:
        try:
            df = pd.read_csv(self.data_transformation_artifact.final_features_path)
            logger.info(f"Loaded transformed data — {len(df):,} rows · {df.shape[1]} columns")
            return df
        except Exception as e:
            raise WeatherException(e, sys)

    # ── Check 1: Required columns ────────────────────────────────
    def validate_columns(self, df: pd.DataFrame) -> bool:
        try:
            required = list(self.schema["columns"].keys())
            missing  = [c for c in required if c not in df.columns]

            if missing:
                self.validation_errors.append(f"Missing columns: {missing}")
                logger.error(f"Missing columns: {missing}")
                return False

            logger.info("Column validation: PASSED")
            return True
        except Exception as e:
            raise WeatherException(e, sys)

    # ── Check 2: Missing values ──────────────────────────────────
    def validate_missing_values(self, df: pd.DataFrame) -> bool:
        try:
            null_counts     = df.isnull().sum()
            cols_with_nulls = null_counts[null_counts > 0]

            if len(cols_with_nulls) > 0:
                self.validation_errors.append(
                    f"Null values found: {cols_with_nulls.to_dict()}"
                )
                logger.warning(f"Null values:\n{cols_with_nulls}")
                return False

            logger.info("Missing value validation: PASSED")
            return True
        except Exception as e:
            raise WeatherException(e, sys)

    # ── Check 3: Value ranges ────────────────────────────────────
    def validate_value_ranges(self, df: pd.DataFrame) -> bool:
        try:
            valid_ranges = self.schema["valid_ranges"]
            range_errors = {}

            for col, bounds in valid_ranges.items():
                if col not in df.columns:
                    continue
                lo, hi     = bounds["min"], bounds["max"]
                violations = ((df[col] < lo) | (df[col] > hi)).sum()
                if violations > 0:
                    range_errors[col] = {
                        "violations":  int(violations),
                        "min_allowed": lo,
                        "max_allowed": hi,
                        "actual_min":  round(float(df[col].min()), 4),
                        "actual_max":  round(float(df[col].max()), 4),
                    }
                    logger.warning(f"{col}: {violations} out-of-range values")

            if range_errors:
                self.validation_errors.append(f"Range violations: {range_errors}")
                return False

            logger.info("Range validation: PASSED")
            return True
        except Exception as e:
            raise WeatherException(e, sys)

    # ── Check 4: rain_tomorrow is binary ─────────────────────────
    def validate_target_column(self, df: pd.DataFrame) -> bool:
        try:
            if "rain_tomorrow" not in df.columns:
                self.validation_errors.append("rain_tomorrow column missing")
                return False

            unique_vals = df["rain_tomorrow"].dropna().unique()
            if not all(v in [0, 1] for v in unique_vals):
                self.validation_errors.append(
                    f"rain_tomorrow has non-binary values: {unique_vals}"
                )
                logger.error(f"rain_tomorrow non-binary: {unique_vals}")
                return False

            rain_pct = round(100 * df["rain_tomorrow"].sum() / len(df), 2)
            logger.info(f"Target validation: PASSED — rain_pct={rain_pct}%")
            return True
        except Exception as e:
            raise WeatherException(e, sys)

    # ── Check 5: Temporal continuity ─────────────────────────────
    def validate_temporal_continuity(self, df: pd.DataFrame) -> bool:
        try:
            if "valid_time" not in df.columns:
                logger.warning("valid_time not found — skipping temporal check")
                return True

            df["valid_time"] = pd.to_datetime(df["valid_time"])
            diffs    = df["valid_time"].diff().dropna()
            expected = pd.Timedelta("1h")
            gaps     = diffs[diffs != expected]

            if len(gaps) > 0:
                self.validation_errors.append(
                    f"Temporal gaps found: {len(gaps)} non-1h intervals"
                )
                logger.warning(f"Temporal gaps: {len(gaps)}")
                return False

            logger.info("Temporal continuity: PASSED")
            return True
        except Exception as e:
            raise WeatherException(e, sys)

    # ── Build report ─────────────────────────────────────────────
    def _build_report(self, df: pd.DataFrame, status: bool) -> dict:
        return {
            "validation_status": status,
            "total_rows":        len(df),
            "total_columns":     df.shape[1],
            "columns":           list(df.columns),
            "errors":            self.validation_errors,
            "null_counts":       df.isnull().sum().to_dict(),
        }

    # ── Orchestrator ─────────────────────────────────────────────
    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            logger.info("=" * 60)
            logger.info("Starting Data Validation")
            logger.info("=" * 60)

            df = self._load_data()

            checks = [
                self.validate_columns(df),
                self.validate_missing_values(df),
                self.validate_value_ranges(df),
                self.validate_target_column(df),
                self.validate_temporal_continuity(df),
            ]

            validation_status = all(checks)

            report = self._build_report(df, validation_status)
            create_directories([os.path.dirname(self.config.report_file_path)])
            save_json(self.config.report_file_path, report)

            artifact = DataValidationArtifact(
                validation_status=validation_status,
                report_file_path=self.config.report_file_path,
                message=(
                    "Validation passed." if validation_status
                    else f"Validation failed: {self.validation_errors}"
                ),
            )

            logger.info("=" * 60)
            logger.info(f"Data Validation Artifact: {artifact}")
            logger.info("=" * 60)
            return artifact

        except Exception as e:
            raise WeatherException(e, sys)


if __name__ == "__main__":
    from src.entity.config_entity import (
        DataIngestionConfig,
        DataTransformationConfig,
        DataValidationConfig,
    )
    from src.components.data_ingestion import DataIngestion
    from src.components.data_transformation import DataTransformation

    ingestion_config   = DataIngestionConfig()
    ingestion          = DataIngestion(ingestion_config)
    ingestion_artifact = ingestion.initiate_data_ingestion()
    
    transformation_config   = DataTransformationConfig()
    transformation          = DataTransformation(transformation_config, ingestion_artifact)
    transformation_artifact = transformation.initiate_data_transformation()

    # ingestion_artifact      = DataIngestion(DataIngestionConfig()).initiate_data_ingestion()
    # transformation_artifact = DataTransformation(DataTransformationConfig(), ingestion_artifact).initiate_data_transformation()
    validation_config = DataValidationConfig()
    validation = DataValidation(validation_config, transformation_artifact)

    # validation              = DataValidation(DataValidationConfig(), transformation_artifact)
    validation_artifact                = validation.initiate_data_validation()