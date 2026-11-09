"""Compare saved handwriting transcriptions with their reference text."""

import argparse
import csv
from pathlib import Path


def edit_distance(reference, prediction):
    previous = list(range(len(prediction) + 1))
    for index, reference_item in enumerate(reference, start=1):
        current = [index]
        for position, predicted_item in enumerate(prediction, start=1):
            current.append(min(
                current[-1] + 1,
                previous[position] + 1,
                previous[position - 1] + (reference_item != predicted_item),
            ))
        previous = current
    return previous[-1]


def normalize(text):
    return " ".join(text.casefold().split())


def evaluate(path):
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        columns = reader.fieldnames or []
        if not {"text_pred", "text_true"}.issubset(columns):
            raise ValueError("CSV must contain text_pred and text_true columns")
        variants = {"text_pred": "Recognizer"}
        if "Chat GPT" in columns:
            variants["Chat GPT"] = "After ChatGPT cleanup"

        totals = {column: [0, 0] for column in variants}
        pages = reference_characters = reference_words = 0
        for line, row in enumerate(reader, start=2):
            reference = normalize(row["text_true"] or "")
            if not reference:
                raise ValueError(f"Missing reference text on line {line}")
            reference_tokens = reference.split()
            reference_characters += len(reference)
            reference_words += len(reference_tokens)
            pages += 1
            for column in variants:
                if row[column] is None:
                    raise ValueError(f"Missing {column} field on line {line}")
                prediction = normalize(row[column])
                totals[column][0] += edit_distance(reference, prediction)
                totals[column][1] += edit_distance(reference_tokens, prediction.split())

    if not pages:
        raise ValueError("CSV contains no transcriptions")

    print(f"{pages} pages, {reference_characters} reference characters, "
          f"{reference_words} reference words")
    for column, label in variants.items():
        character_edits, word_edits = totals[column]
        print(f"{label}: CER {character_edits / reference_characters:.2%}, "
              f"WER {word_edits / reference_words:.2%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, nargs="?",
                        default=Path(__file__).resolve().parents[2]
                        / "outputs/model_predictions_chatGPT_responses.csv")
    args = parser.parse_args()
    try:
        evaluate(args.csv)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")
