"""Launch word training in the project's persistent .venv.

Example: python train.py --words-dir C:/datasets/IAM/words --epochs 50
All arguments are forwarded to scripts/training/CTC_Word.py.
"""
from pathlib import Path
import os
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parent
    environment = root / ".venv"
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not python.is_file():
        print("Project environment is missing. Run: powershell -File setup_env.ps1", file=sys.stderr)
        return 1
    return subprocess.call([
        str(python), str(root / "scripts/training/CTC_Word.py"), *sys.argv[1:]
    ])


if __name__ == "__main__":
    raise SystemExit(main())
