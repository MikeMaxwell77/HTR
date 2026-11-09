# Handwritten Text Recognition

This project transcribes IAM handwriting pages. OpenCV detects word regions and orders them by reading position. A saved Keras recognizer decodes the normalized word crops using CTC, then the page pipeline writes the resulting text to CSV.

![IAM handwriting page with detected word boxes numbered in reading order](outputs/Detected%20Words%20Sorted%20in%20Red.png)

## Evaluate the saved demo

Run from the repository root with Python. This evaluates the checked-in transcripts and requires no model, GPU, or API key.

```bash
python scripts/evaluation/evaluate_transcriptions.py
```

| Output on 22 archived IAM pages | Character error rate | Word error rate |
| --- | ---: | ---: |
| Recognizer | 66.98% | 90.56% |
| After ChatGPT cleanup | 61.51% | 74.08% |

The script totals edit distances across pages and divides by the total reference character or word count. It folds case, collapses whitespace, and retains punctuation. The cleanup improved the saved outputs by 16.48 percentage points in word error rate. The remaining errors are substantial, and these pages provide a demo comparison rather than a writer-disjoint benchmark.

## Transcribe IAM pages

Run from the repository root with the inference dependencies installed. Supply IAM PNG files named by form ID and a reference CSV with `Form ID` and `Form Data` columns. The saved recognizer lives at `models/pred.keras`.

```powershell
python scripts/HTR_Main.py --forms-dir "C:/datasets/IAM/Forms" --forms-csv "C:/datasets/IAM/forms_data.csv" --no-plots
```

The inference script uses NumPy, OpenCV, Keras, TensorFlow, Matplotlib, pandas, and TextBlob, with PyTorch as the default Keras backend. It writes `outputs/model_predictions.csv`. Score that file with `python scripts/evaluation/evaluate_transcriptions.py outputs/model_predictions.csv` once it exists. The exported `text_pred` column contains the recognizer output before the optional cleanup stage.

To build the reference CSV from the IAM annotations, run:

```powershell
python "scripts/data_preparation/Make formsCSV.py" "C:/datasets/IAM/ascii/lines.txt"
```

The generator writes `data/forms_data.csv` by default and keeps the original demo’s
filter for form IDs beginning with `j`. Pass that path as `--forms-csv` for inference.

The main modules live in `scripts/segmentation/`. The detection, crop extraction, recognition, and export steps are separate so errors can be traced to the relevant stage.

## Keras CTC word training

For RTX 5050 GPU training in Docker and a portable image for future AWS use,
see [the Docker guide](docs/DOCKER.md). Quick start after installing Docker Desktop
with its WSL 2 backend:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\docker.ps1 Setup
powershell -NoProfile -ExecutionPolicy Bypass -File .\docker.ps1 Smoke
powershell -NoProfile -ExecutionPolicy Bypass -File .\docker.ps1 Train --batch-size 2
```

The launcher locates Docker Desktop automatically and rebuilds when needed.
`Setup` verifies GPU training; `Smoke` runs one epoch on 100 images and writes to
a separate model folder; `Train` uses the settings in `train.py` plus overrides.

`scripts/training/CTC_Word.py` trains three convolutional blocks followed by
exactly two bidirectional LSTMs (128 then 64 units per direction), and a character
logit head. Keras CTC loss and greedy decoding both use blank index 0. Labels
retain case, digits, and punctuation. See the [Keras CTC API](https://keras.io/api/ops/nn/#ctc_loss-function).

```powershell
python -m pip install -r scripts/training/requirements.txt
python scripts/training/CTC_Word.py --words-dir "C:/datasets/IAM/words"
```

The CSV requires `image_path` (relative to `--words-dir`) and `target` columns.
Use `--limit 100 --epochs 1` for a small trial. Images are loaded in batches.
The seeded 80/20 split is by image, not by writer; validation accuracy does not
measure generalization to unseen writers.

Preprocessing accepts grayscale images and resizes them with LANCZOS to fit a
128 x 32 canvas while preserving aspect ratio, then pads white and rotates exactly
90 degrees clockwise. Override the canvas with `--width` and `--height` (before
rotation). No full image scan is needed at startup. The default width grows when
long labels require more CTC timesteps (420 for the current full CSV). An explicit
width that is too small causes a clear error instead of invalid training loss.
`--png-dir outputs/standardized_words` optionally exports the resized images as
lossless PNGs with compression level 9. Resizing changes the original pixels;
PNG compression preserves the resized result exactly. Model inputs are normalized
to [0, 1] after rotation. Each bidirectional LSTM is followed by 15% dropout; the
existing 20% dropout before the LSTMs remains. Saved metadata records resizing
and dropout. Earlier models require their original preprocessing.

Outputs go to `models/ctc_word/`: `word_recognition.keras`, `metadata.json`,
`history.json`, and `validation_predictions.csv`. Load the model with
`keras.models.load_model`, preprocess new images with
`scripts.training.word_preprocessing.preprocess_image(path, width, height)` using
the saved dimensions, and decode logits with
`scripts.training.CTC_Word.decode_batch_predictions(logits, characters)` using the
saved vocabulary. This model has a different preprocessing/vocabulary contract
from the existing segmentation model; it is not a drop-in replacement for `pred.keras`.

Full checkpoints now save after every epoch (latest five retained), alongside
`best.keras` for the lowest validation loss. Use `--resume` to continue with the
saved optimizer state and completed epoch; use a new output directory for changed
data/model settings. For background GPU training with automatic resume, a target
of 1,000 epochs and early stopping disabled:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\docker.ps1 Continuous
```

Keep the computer awake and Docker running. See [checkpoint and background run
instructions](docs/DOCKER.md#background-training-with-epoch-checkpoints) for logs,
stopping, retention and saved file locations.

Run checks with `python -m unittest discover -s tests -v`. The model training,
decoding, and save/load smoke test requires TensorFlow and Keras.

## Persistent Python environment and training launcher

The project uses a local `.venv` with Python 3.12. It persists between terminal
and editor sessions and is excluded from Git. To recreate or refresh it on Windows:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_env.ps1
```

The setup script installs the exact versions in
`scripts/training/requirements-lock.txt` (captured on Windows with Python 3.12).
The broader dependency constraints remain in `scripts/training/requirements.txt`.

Set `WORDS_DIR` at the top of `train.py` to the extracted IAM words folder
(default: `data/IAM_Words/words`). Set epochs, batch size, and optional image limit there
as well. Then run the file in your IDE or from the repository root; activation
and command-line options are not required:

```powershell
python train.py
```

`train.py` passes the saved settings to the CTC trainer using `.venv`'s interpreter.
Relative dataset paths are resolved from the project root.
Use `python train.py --help` for all options. Outputs default to `models/ctc_word/`.
For IDE runs, select `.venv/Scripts/python.exe` as the Python interpreter.
To run the checks with this environment:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```
