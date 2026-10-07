"""
Data Ingestion Module

This module handles the complete ERA5 weather data ingestion pipeline:
1. Downloads ERA5 data from the Copernicus Climate Data Store (CDS).
2. Extracts downloaded archives.
3. Loads and merges NetCDF datasets.
4. Filters data for the target geographical location.
5. Builds and saves the raw weather dataset as a CSV.
6. Returns a DataIngestionArtifact for downstream MLOps pipeline stages.
"""
import os
import sys
import zipfile
from pathlib import Path

import cdsapi
import pandas as pd
import xarray as xr
from dotenv import load_dotenv

from src.logger import logger
from src.exception import WeatherException
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
from src.utils.main_utils import create_directories, read_yaml_file


MONTH_GROUPS = {
    "JAN": "01",
    "FEB": "02",
    "MAR": "03",
    "APR": "04",
    "MAY": "05",
    "JUN": "06",
    "JUL": "07",
    "AUG": "08",
    "SEP": "09",
    "OCT": "10",
    "NOV": "11",
    "DEC": "12",
}


class DataIngestion:

    def __init__(self, config: DataIngestionConfig):
        self.config = config

        load_dotenv()

        self.params = read_yaml_file("params.yaml")

        self.years = self.params["data"]["years"]
        self.area = self.params["data"]["area"]
        self.variables = self.params["data"]["variables"]
        self.target_lat = self.params["data"]["target_latitude"]
        self.target_lon = self.params["data"]["target_longitude"]

    # ---------------------------------------------------------
    # ERA5 CLIENT
    # ---------------------------------------------------------

    def _get_cds_client(self):
        try:
            cds_url = os.getenv("CDS_API_URL")
            cds_key = os.getenv("CDS_API_KEY")

            if not cds_url or not cds_key:
                raise ValueError("CDS_API_URL and CDS_API_KEY must be set in .env")

            return cdsapi.Client(
                url=cds_url,
                key=cds_key,
            )

        except Exception as e:
            raise WeatherException(e, sys)

    # ---------------------------------------------------------
    # PATH HELPERS
    # ---------------------------------------------------------

    def _get_archive_path(self, year, month_name):
        return (
            Path(self.config.archive_dir)
            / str(year)
            / f"azamgarh_weather_{month_name}_{year}.zip"
        )

    def _get_raw_month_path(self, year, month_name):
        return (
            Path(self.config.raw_dir)
            / str(year)
            / month_name
        )

    # ---------------------------------------------------------
    # STEP 1: DOWNLOAD ERA5
    # ---------------------------------------------------------

    def download_era5_data(self) -> None:
        try:
            client = self._get_cds_client()

            days = [f"{day:02d}" for day in range(1, 32)]
            times = [f"{hour:02d}:00" for hour in range(24)]

            for year in self.years:

                for month_name, month in MONTH_GROUPS.items():

                    archive_path = self._get_archive_path(year, month_name)

                    if archive_path.exists():
                        logger.info(f"Already exists, skipping -> {archive_path.name}")
                        continue

                    archive_path.parent.mkdir(parents=True, exist_ok=True)

                    logger.info(f"Downloading {month_name}_{year}...")

                    client.retrieve(
                        "reanalysis-era5-single-levels",
                        {
                            "product_type": "reanalysis",
                            "variable": self.variables,
                            "year": str(year),
                            "month": month,
                            "day": days,
                            "time": times,
                            "area": self.area,
                            "format": "netcdf",
                        },
                        str(archive_path),
                    )

                    logger.info(f"Downloaded -> {archive_path.name}")

        except Exception as e:
            raise WeatherException(e, sys)

    # ---------------------------------------------------------
    # STEP 2: EXTRACT ARCHIVES
    # ---------------------------------------------------------

    def extract_archives(self) -> None:
        try:
            for year in self.years:

                for month_name in MONTH_GROUPS:

                    zip_path = self._get_archive_path(year, month_name)

                    extract_path = self._get_raw_month_path(year, month_name)

                    if extract_path.exists() and any(extract_path.iterdir()):
                        logger.info(f"Already extracted, skipping -> {month_name}_{year}")
                        continue

                    if not zip_path.exists():
                        raise FileNotFoundError(f"Archive not found: {zip_path}")

                    extract_path.mkdir(parents=True, exist_ok=True)

                    with zipfile.ZipFile(zip_path, "r") as zf:
                        zf.extractall(extract_path)

                    logger.info(f"Extracted -> {month_name}_{year}")

        except Exception as e:
            raise WeatherException(e, sys)


    # ---------------------------------------------------------
    # STEP 3: LOAD NETCDF DATA
    # ---------------------------------------------------------

    def _load_netcdf(self, file_path):
        with xr.open_dataset(file_path, engine="netcdf4") as ds:
            return ds.to_dataframe().reset_index()

    # ---------------------------------------------------------
    # STEP 4: BUILD RAW DATASET
    # ---------------------------------------------------------

    def build_raw_csv(self) -> pd.DataFrame:
        try:
            instant_dfs = []
            precipitation_dfs = []

            for year in self.years:

                for month_name in MONTH_GROUPS:

                    base_path = self._get_raw_month_path(year, month_name)

                    instant_file = (
                        base_path
                        / "data_stream-oper_stepType-instant.nc"
                    )

                    accum_file = (
                        base_path
                        / "data_stream-oper_stepType-accum.nc"
                    )

                    instant_dfs.append(self._load_netcdf(instant_file))

                    precipitation_dfs.append(self._load_netcdf(accum_file))

            instant_df = pd.concat(instant_dfs, ignore_index=True)

            precipitation_df = pd.concat(precipitation_dfs, ignore_index=True)

            # Filter target location
            instant_df = instant_df[
                (instant_df["latitude"] == self.target_lat)
                & (instant_df["longitude"] == self.target_lon)
            ].reset_index(drop=True)

            precipitation_df = precipitation_df[
                (precipitation_df["latitude"] == self.target_lat)
                & (precipitation_df["longitude"] == self.target_lon)
            ].reset_index(drop=True)

            # Add precipitation
            if len(instant_df) != len(precipitation_df):
                raise ValueError("Instant and precipitation datasets have different row counts.")

            instant_df["tp"] = precipitation_df["tp"].values

            logger.info(f"Raw dataset built — {len(instant_df):,} rows")

            return instant_df

        except Exception as e:
            raise WeatherException(e, sys)


    # ---------------------------------------------------------
    # STEP 5: SAVE RAW DATA
    # ---------------------------------------------------------

    def save_raw_data(self, df: pd.DataFrame) -> None:
        try:
            create_directories([
                os.path.dirname(self.config.raw_file_path)
            ])

            df.to_csv(self.config.raw_file_path, index=False)

            logger.info(f"Raw CSV saved -> {self.config.raw_file_path}")

        except Exception as e:
            raise WeatherException(e, sys)


    # ---------------------------------------------------------
    # ORCHESTRATOR
    # ---------------------------------------------------------

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:

            logger.info("=" * 60)
            logger.info("Starting Data Ingestion — ERA5")
            logger.info("=" * 60)

            self.download_era5_data()
            self.extract_archives()

            df = self.build_raw_csv()

            self.save_raw_data(df)

            artifact = DataIngestionArtifact(
                raw_file_path=self.config.raw_file_path,
                is_ingested=True,
                message=f"ERA5 ingestion complete. {len(df):,} rows saved."
            )

            logger.info("=" * 60)
            logger.info(f"Data Ingestion Artifact: {artifact}")
            logger.info("=" * 60)

            return artifact

        except Exception as e:
            raise WeatherException(e, sys)


if __name__ == "__main__":
    from src.entity.config_entity import DataIngestionConfig

    config   = DataIngestionConfig()
    ingestion = DataIngestion(config)
    artifact = ingestion.initiate_data_ingestion()