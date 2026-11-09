"""Build reference transcriptions for the IAM page demo."""

import argparse
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def build_forms_csv(lines_path, output_path):
    forms = {}
    with lines_path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = line.split(maxsplit=8)
            if len(parts) < 9 or parts[1] != "ok":
                continue
            form_id = "-".join(parts[0].split("-")[:2])
            if form_id.startswith("j"):
                text = parts[8].replace("|", " ").strip()
                forms.setdefault(form_id, []).append(text)

    if not forms:
        raise ValueError("No usable IAM annotations for forms beginning with j")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["Form ID", "Form Data"])
        writer.writerows((form_id, " ".join(lines)) for form_id, lines in forms.items())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lines_txt", type=Path, help="Path to IAM ascii/lines.txt")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data/forms_data.csv")
    args = parser.parse_args()
    try:
        build_forms_csv(args.lines_txt, args.output)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")
