"""Default input and output locations."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models/pred.keras"
FORMS_CSV = PROJECT_ROOT / "data/forms_data.csv"
OUTPUT_CSV = PROJECT_ROOT / "outputs/model_predictions.csv"
FORMS_DIR = Path("C:/Users/mikey/OneDrive/Documents/USCB/Data Mining/Project/Forms")
