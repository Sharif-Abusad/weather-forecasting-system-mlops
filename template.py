"""
template.py — Run once from project root to scaffold the project.
    python template.py
"""

from pathlib import Path

DIRS = [
    # App (FastAPI)
    "app/routers",
    "app",

    # Source — pipeline components (mirrors your existing project)
    "src/components",
    "src/configuration",
    "src/constants",
    "src/data_access",
    "src/entity",
    "src/exception",
    "src/logger",
    "src/pipeline",
    "src/utils",

    # Config files
    "config",

    # Models & artifacts
    "artifact",

    # DVC tracked data
    "data/raw",
    "data/processed",
    "data/features",

    # MLflow
    "mlruns",

    # Reports & plots
    "reports/figures",

    # Logs
    "logs",

    # Notebooks
    "notebook",

    # Frontend
    "static",
    "templates",
]

FILES = {
    # ── Root files ──────────────────────────────────────────
    "main.py": "",
    "demo.py": "",
    "setup.py": "",
    "requirements.txt": "",
    "Dockerfile": "",
    ".dockerignore": "",
    ".env": "",
    ".gitignore": "",
    "README.md": "",
    "pyproject.toml": "",
    "projectflow.txt": "",

    # ── Config YAMLs ─────────────────────────────────────────
    "config/model.yaml": "",
    "config/schema.yaml": "",

    # ── DVC + params ─────────────────────────────────────────
    "params.yaml": "",
    "dvc.yaml": "",

    # ── App ──────────────────────────────────────────────────
    "app/__init__.py": "",
    "app/routers/__init__.py": "",
    "app/routers/weather_routes.py": "",

    # ── src root ─────────────────────────────────────────────
    "src/__init__.py": "",

    # ── components (one file per pipeline stage) ─────────────
    "src/components/__init__.py": "",
    "src/components/data_ingestion.py": "",
    "src/components/data_transformation.py": "",
    "src/components/data_validation.py": "",
    "src/components/model_trainer.py": "",
    "src/components/model_evaluation.py": "",
    "src/components/model_pusher.py": "",

    # ── configuration (reads config/model.yaml & schema.yaml) ─
    "src/configuration/__init__.py": "",
    "src/configuration/mongo_db_connection.py": "",   # swap to local if no mongo

    # ── constants ────────────────────────────────────────────
    "src/constants/__init__.py": "",

    # ── data_access ──────────────────────────────────────────
    "src/data_access/__init__.py": "",
    "src/data_access/weather_data.py": "",

    # ── entity (config + artifact dataclasses) ───────────────
    "src/entity/__init__.py": "",
    "src/entity/config_entity.py": "",
    "src/entity/artifact_entity.py": "",
    "src/entity/estimator.py": "",

    # ── exception ────────────────────────────────────────────
    "src/exception/__init__.py": "",

    # ── logger ───────────────────────────────────────────────
    "src/logger/__init__.py": "",

    # ── pipeline ─────────────────────────────────────────────
    "src/pipeline/__init__.py": "",
    "src/pipeline/training_pipeline.py": "",
    "src/pipeline/prediction_pipeline.py": "",

    # ── utils ────────────────────────────────────────────────
    "src/utils/__init__.py": "",
    "src/utils/main_utils.py": "",

    # ── notebook ─────────────────────────────────────────────
    "notebook/exp-notebook.ipynb": "{}",
}


def create_structure():
    root = Path(".")

    for d in DIRS:
        path = root / d
        path.mkdir(parents=True, exist_ok=True)
        gitkeep = path / ".gitkeep"
        if not list(path.iterdir()) and not gitkeep.exists():
            gitkeep.touch()

    created = skipped = 0
    for rel, content in FILES.items():
        fp = root / rel
        fp.parent.mkdir(parents=True, exist_ok=True)
        if fp.exists():
            skipped += 1
        else:
            fp.write_text(content, encoding="utf-8")
            created += 1

    print(f"\n✅  Scaffold done — {created} created, {skipped} skipped")
    print("\nNext → Step 2: fill in src/logger/__init__.py")


if __name__ == "__main__":
    create_structure()