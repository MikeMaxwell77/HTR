"""Coordinate detection, recognition, and CSV export without running on import."""
import cv2
import numpy as np
import pandas as pd
from .config import FORMS_CSV, FORMS_DIR, MODEL_PATH, OUTPUT_CSV
from .detection import RegionFocusedTextDetector
from .extraction import sort_contours, extract_words
from .recognition import load_recognition_model, decode_batch_predictions
from .text_processing import assemble_sentence, clean_text, compare_strings
from .visualization import show_extracted_words
from pathlib import Path


def run_pipeline(forms_dir=FORMS_DIR, forms_csv=FORMS_CSV,
                 model_path=MODEL_PATH, output_csv=OUTPUT_CSV, show_plots=True):
    """Process IAM forms and return the exported prediction DataFrame."""
    forms_data = pd.read_csv(forms_csv)
    model = load_recognition_model(model_path)
    detector = RegionFocusedTextDetector()
    records = []
    for _, row in forms_data.iterrows():
        form_id, form_info = row['Form ID'], row['Form Data']
        sample_path = str(Path(forms_dir) / f"{form_id}.png")
        image = cv2.imread(sample_path)
        if image is None:
            print(f"Error: Could not read image at {sample_path}")
            continue
        _, contours = detector.detect_text_regions(sample_path)
        print(f"Detected {len(contours)} potential text regions.")
        sorted_contours, line_count = sort_contours(contours)
        print(f"Contours sorted into {line_count} lines.")
        words, boxes = extract_words(image, sorted_contours)
        if not words:
            print(f"Error: no valid word regions extracted for {form_id}.")
            continue
        predictions = model.predict(np.array(words, dtype=np.float32))
        texts = decode_batch_predictions(predictions)
        print("\n--- Predicted Text (Sorted Order) ---")
        for i, (text, box) in enumerate(zip(texts, boxes), 1):
            print(f"word {i} (box: {box}): {text}")
        if show_plots:
            show_extracted_words(image, boxes, len(contours))
        sentence = assemble_sentence(texts)
        cleaned = clean_text(sentence)
        print(sentence)
        print(cleaned)
        common = compare_strings(cleaned, form_info)
        # Preserve v5's character-length denominators; these are not WER scores.
        print(f"Percent of words from the form in common{common / len(form_info) if form_info else 0}")
        print(f"Percent of words from the predictions in common{common / len(cleaned) if cleaned else 0}")
        records.append([form_id, sentence, form_info])
    results = pd.DataFrame(records, columns=['form_id', 'text_pred', 'text_true'])
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_csv, index=True)
    return results
