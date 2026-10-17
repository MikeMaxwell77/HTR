# HTR

IAM handwriting recognition, centered on `scripts/Better_Segmentation2Keras_v5.py`.
See [the file guide](docs/FILE_GUIDE.md) for every file and its role.

## Run Better Segmentation v5

```powershell
python scripts/Better_Segmentation2Keras_v5.py
```

It requires `models/pred.keras`, `data/forms_data.csv`, and IAM form PNGs.
Pass your IAM forms directory using `--forms-dir "C:/path/to/Forms"`.
The original machine-specific default is retained in `scripts/segmentation/config.py`. The images are not included. Required third-party imports include NumPy,
OpenCV, Keras, TensorFlow, Matplotlib, pandas, and TextBlob.

No other project script needs to run first when the existing model and CSV are
available. Results are written to `outputs/model_predictions.csv`; plots and
console output are also produced. The exported text is the raw prediction;
TextBlob-corrected text is printed but is not the exported `text_pred` value.

## Organization

- `scripts/Better_Segmentation2Keras_v5.py`: main inference workflow.
- `scripts/data_preparation/`: annotation-to-CSV generators.
- `scripts/training/`: separate model training experiments.
- `scripts/evaluation/`: evaluation of existing ChatGPT-refined results.
- `scripts/examples/`: standalone image-loading and API examples.
- `scripts/archive/`: unnecessary RNN experiment and duplicate generators.
- `data/`: annotation CSVs.
- `models/`: saved model and weights.
- `outputs/`: prediction CSVs and visualizations.
- `docs/`: file guide and demo link.

Run scripts from the repository root. Some auxiliary scripts retain original
absolute IAM dataset paths or relative `words/` paths; configure them locally.
The training source `human_text_recognition.py` contains notebook shell commands
and is not directly runnable with Python. Model-training scripts save different
filenames from `pred.keras`; they are not an automatic rebuild pipeline for it.

## Segmentation modules

The v5 script is now a CLI entry point. Its implementation lives in
`scripts/segmentation/`:

| Module | Responsibility |
| --- | --- |
| `config.py` | Default dataset, model, and output paths |
| `detection.py` | OpenCV region detection and detector debug views |
| `extraction.py` | Reading-order sorting, padded crops, and alternate segmentation helper |
| `preprocessing.py` | Aspect-ratio resize, padding, transpose, and normalization |
| `recognition.py` | Model loading, character vocabulary, and CTC decoding |
| `text_processing.py` | Header/footer removal, sentence assembly, spelling correction, overlap counts |
| `visualization.py` | Numbered word boxes in reading order |
| `pipeline.py` | Coordinates form processing and exports the prediction CSV |
| `__init__.py` | Marks the reusable Python package |

```powershell
python scripts/Better_Segmentation2Keras_v5.py --forms-dir "C:/datasets/IAM/Forms" --no-plots
```

Optional overrides: `--forms-csv`, `--model`, and `--output`. Importing the
entry point does not start inference. Missing images and forms with no valid
word crops are reported and skipped. The existing header/footer convention
and overlap-score formulas remain; those overlap scores are not word error rate.

## Keras CTC word training

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

Preprocessing accepts grayscale images and preserves their pixels: white padding
to a common canvas, then an exact 90-degree clockwise rotation. No resizing or
cropping occurs. Canvas dimensions are inferred from the selected images and CTC
length requirements, or supplied with `--width` and `--height` (before rotation).
Oversized images cause an error instead of silent downsampling. A large source
image can therefore make every training tensor large; reduce batch size if needed.
Lossless compression reduces file size, not image dimensions or model memory.
`--png-dir data/standardized_words` optionally exports lossless PNGs. Model inputs
are normalized to [0, 1] after rotation.

Outputs go to `models/ctc_word/`: `word_recognition.keras`, `metadata.json`,
`history.json`, and `validation_predictions.csv`. Load the model with
`keras.models.load_model`, preprocess new images with
`scripts.training.word_preprocessing.preprocess_image(path, width, height)` using
the saved dimensions, and decode logits with
`scripts.training.CTC_Word.decode_batch_predictions(logits, characters)` using the
saved vocabulary. This model has a different preprocessing/vocabulary contract
from the existing segmentation model; it is not a drop-in replacement for `pred.keras`.

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
(default: `data/words`). Set epochs, batch size, and optional image limit there
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
