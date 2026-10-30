"""Launch word training in the project's .venv or Docker interpreter.

Edit the settings below, then run this file from your IDE or use python train.py.
"""
from pathlib import Path
import os
import subprocess
import sys

# Training settings: set WORDS_DIR to your IAM words folder once.
PROJECT_ROOT = Path(__file__).resolve().parent
WORDS_DIR = PROJECT_ROOT / "data" / "IAM_Words" / "words"
CSV_PATH = PROJECT_ROOT / "data" / "word_dataset_info.csv"
OUTPUT_DIR = PROJECT_ROOT / "models" / "ctc_word"
EPOCHS = 50
BATCH_SIZE = 8
LSTM_UNITS = 128  # Second layer automatically uses half this many units.
LIMIT = None  # Set to 100 for a quick trial; None trains on the full dataset.


def main():
    root = Path(__file__).resolve().parent
    environment = root / ".venv"
    python = (Path(sys.executable) if os.environ.get("HTR_CONTAINER") == "1" else
              environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python"))
    if not python.is_file():
        print("Project environment is missing. Run: powershell -File setup_env.ps1", file=sys.stderr)
        return 1
    arguments = [
        "--words-dir", str(WORDS_DIR), "--csv", str(CSV_PATH),
        "--output-dir", str(OUTPUT_DIR), "--epochs", str(EPOCHS),
        "--batch-size", str(BATCH_SIZE), "--lstm-units", str(LSTM_UNITS),
    ]
    if LIMIT is not None:
        arguments.extend(["--limit", str(LIMIT)])
    if not sys.argv[1:] and not WORDS_DIR.is_dir():
        print(f"Set WORDS_DIR at the top of train.py to your IAM words folder.\n"
              f"Current location does not exist: {WORDS_DIR}", file=sys.stderr)
        return 1
    command = [
        str(python), str(root / "scripts/training/CTC_Word.py"),
        *arguments, *sys.argv[1:]
    ]
    if os.environ.get("HTR_CONTAINER") == "1":
        os.chdir(root)
        # Replace the launcher so container termination reaches the trainer.
        os.execv(str(python), command)
    return subprocess.call(command, cwd=root)


if __name__ == "__main__":
    raise SystemExit(main())
