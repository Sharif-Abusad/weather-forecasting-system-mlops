"""
Data Transformation Module

Transforms raw ERA5 weather data into ML-ready features:
1. Renames ERA5 variable names to model-friendly names.
2. Converts units (Kelvin -> °C, Pa -> hPa, metres -> mm).
3. Computes humidity from dewpoint and wind speed from u/v components.
4. Sorts, deduplicates and checks temporal continuity.
5. Handles missing values via linear interpolation.
6. Clips out-of-range values using schema valid ranges.
7. Engineers temporal, lag, rolling and cyclic features.
8. Creates rain_tomorrow binary target.
9. Saves cleaned CSV and final feature CSV.
10. Returns a DataTransformationArtifact for downstream stages.
"""

import os
import sys
import numpy as np
import pandas as pd
from src.logger import logger
from src.exception import WeatherException
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact
from src.utils.main_utils import read_yaml_file, create_directories


# ── ERA5 column rename map ────────────────────────────────────
RENAME_MAP = {
    "t2m": "temperature_raw",
    "sp":  "surface_pressure_raw",
    "tcc": "total_cloud_cover",
    "lcc": "low_cloud_cover",
    "mcc": "medium_cloud_cover",
    "hcc": "high_cloud_cover",
    "tp":  "precipitation_raw",
    "d2m": "dewpoint_raw",
    "u10": "u_wind",
    "v10": "v_wind",
}


class DataTransformation:

    def __init__(
        self,
        config: DataTransformationConfig,
        data_ingestion_artifact: DataIngestionArtifact,
    ):
        self.config = config
        self.data_ingestion_artifact = data_ingestion_artifact
        self.params = read_yaml_file("params.yaml")
        self.schema = read_yaml_file("config/schema.yaml")

    # ── Load ─────────────────────────────────────────────────────
    def _load_data(self) -> pd.DataFrame:
        try:
            df = pd.read_csv(self.data_ingestion_artifact.raw_file_path)
            logger.info(f"Loaded raw data — {len(df):,} rows")
            return df
        except Exception as e:
            raise WeatherException(e, sys)

    # ── Step 1: Rename ───────────────────────────────────────────
    def rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            df.columns = df.columns.str.strip().str.lower()
            df = df.rename(
                columns={k: v for k, v in RENAME_MAP.items() if k in df.columns}
            )
            logger.info("Columns renamed from ERA5 format")
            return df
        except Exception as e:
            raise WeatherException(e, sys)

    # ── Step 2: Parse datetime ───────────────────────────────────
    def parse_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            time_col = next(
                (c for c in df.columns if "time" in c or "date" in c), None
            )
            if time_col is None:
                raise ValueError("No datetime column found")

            df["valid_time"] = pd.to_datetime(df[time_col], utc=True)
            df["valid_time"] = (
                df["valid_time"]
                .dt.tz_convert("Asia/Kolkata")
                .dt.tz_localize(None)
            )
            if time_col != "valid_time":
                df = df.drop(columns=[time_col])

            logger.info(
                f"Date range: {df['valid_time'].min()} -> {df['valid_time'].max()}"
            )
            return df
        except Exception as e:
            raise WeatherException(e, sys)

    # ── Step 3: Unit conversions ─────────────────────────────────
    def unit_conversions(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            # Kelvin -> Celsius
            if "temperature_raw" in df.columns:
                df["temperature"] = (df["temperature_raw"] - 273.15).round(4)
                df = df.drop(columns=["temperature_raw"])

            # Pa -> hPa
            if "surface_pressure_raw" in df.columns:
                df["surface_pressure"] = (df["surface_pressure_raw"] / 100).round(2)
                df = df.drop(columns=["surface_pressure_raw"])

            # Metres -> mm
            if "precipitation_raw" in df.columns:
                df["precipitation"] = (df["precipitation_raw"] * 1000).round(4).clip(lower=0)
                df = df.drop(columns=["precipitation_raw"])

            # Humidity from dewpoint
            if "dewpoint_raw" in df.columns and "temperature" in df.columns:
                td_c = (df["dewpoint_raw"] - 273.15)
                t_c  = df["temperature"]
                df["humidity"] = (
                    100 *
                    np.exp(17.625 * td_c / (243.04 + td_c)) /
                    np.exp(17.625 * t_c  / (243.04 + t_c))
                ).clip(0, 100).round(2)
                df = df.drop(columns=["dewpoint_raw"])

            # Wind speed from u/v components
            if "u_wind" in df.columns and "v_wind" in df.columns:
                df["wind_speed"] = (
                    np.sqrt(df["u_wind"] ** 2 + df["v_wind"] ** 2)
                ).round(4)
                df = df.drop(columns=["u_wind", "v_wind"])

            logger.info("Unit conversions complete")
            return df
        except Exception as e:
            raise WeatherException(e, sys)

    # ── Step 4: Sort + deduplicate ───────────────────────────────
    def sort_and_deduplicate(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            before = len(df)
            df = df.sort_values("valid_time").drop_duplicates(
                subset="valid_time", keep="first"
            ).reset_index(drop=True)
            dropped = before - len(df)
            logger.info(
                f"Deduplicated — dropped {dropped} rows, {len(df):,} remain"
            )
            return df
        except Exception as e:
            raise WeatherException(e, sys)

    # ── Step 5: Handle missing values ────────────────────────────
    def handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            total_nulls  = df[numeric_cols].isnull().sum().sum()

            if total_nulls == 0:
                logger.info("Missing values: none")
                return df

            logger.info(f"Missing values: {total_nulls} — applying linear interpolation")
            for col in numeric_cols:
                if df[col].isnull().sum() > 0:
                    df[col] = df[col].interpolate(
                        method="linear", limit_direction="both"
                    )
            df[numeric_cols] = df[numeric_cols].ffill().bfill()
            return df
        except Exception as e:
            raise WeatherException(e, sys)

    # ── Step 6: Clip out-of-range values ─────────────────────────
    def clip_ranges(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            valid_ranges = self.schema["valid_ranges"]
            for col, bounds in valid_ranges.items():
                if col not in df.columns:
                    continue
                lo, hi = bounds["min"], bounds["max"]
                violations = ((df[col] < lo) | (df[col] > hi)).sum()
                if violations > 0:
                    logger.warning(f"{col}: {violations} values clipped to [{lo}, {hi}]")
                    df[col] = df[col].clip(lower=lo, upper=hi)
            logger.info("Range clipping complete")
            return df
        except Exception as e:
            raise WeatherException(e, sys)

    # ── Step 7: Select clean columns ─────────────────────────────
    def select_clean_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            required = [
                "valid_time", "temperature", "surface_pressure",
                "total_cloud_cover", "low_cloud_cover", "medium_cloud_cover",
                "high_cloud_cover", "precipitation", "humidity", "wind_speed",
            ]
            final = [c for c in required if c in df.columns]
            missing = [c for c in required if c not in df.columns]
            if missing:
                logger.warning(f"Columns not found: {missing}")
            return df[final]
        except Exception as e:
            raise WeatherException(e, sys)

    # ── Step 8: Feature engineering ──────────────────────────────
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            fe = self.params["feature_engineering"]

            # Temporal
            df["hour"]        = df["valid_time"].dt.hour
            df["day"]         = df["valid_time"].dt.day
            df["month"]       = df["valid_time"].dt.month
            df["day_of_week"] = df["valid_time"].dt.dayofweek

            # Cyclic encodings
            df["hour_sin"]  = np.sin(2 * np.pi * df["hour"]  / 24).round(6)
            df["hour_cos"]  = np.cos(2 * np.pi * df["hour"]  / 24).round(6)
            df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12).round(6)
            df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12).round(6)

            # Lag features
            for lag in fe["lag_hours"]:
                df[f"temp_lag_{lag}"] = df["temperature"].shift(lag)

            # Rolling mean
            w = fe["rolling_window"]
            df[f"temp_rolling_{w}"] = (
                df["temperature"]
                .rolling(window=w, min_periods=w)
                .mean()
                .round(4)
            )

            # Rain tomorrow target
            threshold = fe["rain_threshold_mm"]
            lookahead = fe["rain_lookahead_hours"]
            next_precip = sum(
                df["precipitation"].shift(-i) for i in range(1, lookahead + 1)
            )
            df["rain_tomorrow"] = (next_precip > threshold).astype("Int64")

            rain_count  = df["rain_tomorrow"].sum()
            total_valid = df["rain_tomorrow"].notna().sum()
            rain_pct    = 100 * rain_count / total_valid if total_valid else 0
            logger.info(
                f"rain_tomorrow — Rain=1: {rain_count:,} ({rain_pct:.1f}%), "
                f"Rain=0: {total_valid - rain_count:,} ({100 - rain_pct:.1f}%)"
            )

            # Drop NaN rows from window edges
            feature_cols = [
                "temperature", "surface_pressure", "total_cloud_cover",
                "low_cloud_cover", "medium_cloud_cover", "high_cloud_cover",
                "precipitation", "humidity", "wind_speed",
                f"temp_rolling_{w}", "temp_lag_1", "temp_lag_24", "rain_tomorrow",
            ]
            existing = [c for c in feature_cols if c in df.columns]
            before   = len(df)
            df       = df.dropna(subset=existing).reset_index(drop=True)
            logger.info(f"Dropped {before - len(df)} NaN rows -> {len(df):,} remain")

            logger.info("Feature engineering complete")
            return df
        except Exception as e:
            raise WeatherException(e, sys)

    # ── Orchestrator ─────────────────────────────────────────────
    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logger.info("=" * 60)
            logger.info("Starting Data Transformation")
            logger.info("=" * 60)

            df = self._load_data()
            df = self.rename_columns(df)
            df = self.parse_datetime(df)
            df = self.unit_conversions(df)
            df = self.sort_and_deduplicate(df)
            df = self.handle_missing(df)
            df = self.clip_ranges(df)
            df = self.select_clean_columns(df)

            # Save clean CSV
            create_directories([os.path.dirname(self.config.clean_file_path)])
            df.to_csv(self.config.clean_file_path, index=False)
            logger.info(f"Clean CSV saved -> {self.config.clean_file_path}")

            # Feature engineering
            df = self.engineer_features(df)

            # Save final features CSV
            create_directories([os.path.dirname(self.config.final_features_path)])
            df.to_csv(self.config.final_features_path, index=False)
            logger.info(f"Final features saved -> {self.config.final_features_path}")

            artifact = DataTransformationArtifact(
                clean_file_path=self.config.clean_file_path,
                final_features_path=self.config.final_features_path,
                is_transformed=True,
                message=f"Transformation complete. {len(df):,} rows.",
            )

            logger.info("=" * 60)
            logger.info(f"Data Transformation Artifact: {artifact}")
            logger.info("=" * 60)
            return artifact

        except Exception as e:
            raise WeatherException(e, sys)


if __name__ == "__main__":
    from src.entity.config_entity import (
        DataIngestionConfig,
        DataTransformationConfig
    )
    from src.components.data_ingestion import DataIngestion

    ingestion_config   = DataIngestionConfig()
    ingestion          = DataIngestion(ingestion_config)
    ingestion_artifact = ingestion.initiate_data_ingestion()
    
    transformation_config   = DataTransformationConfig()
    transformation          = DataTransformation(transformation_config, ingestion_artifact)
    transformation_artifact = transformation.initiate_data_transformation()