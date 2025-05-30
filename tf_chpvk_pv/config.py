from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
BANDGAP_DATA_DIR = DATA_DIR / "bandgap_semicon"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODNET_DIR = PROJ_ROOT / "modnet"
CRABNET_DIR = PROJ_ROOT / "crabnet"

MODELS_DIR = PROJ_ROOT / "models"
TREES_DIR = MODELS_DIR / "trees"
RESULTS_DIR = MODELS_DIR/ "results"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
RESULTS_DIR = MODELS_DIR/ "results"

# Random seed
RANDOM_SEED = 187636123

# Primary features to use

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
