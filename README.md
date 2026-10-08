# HTR

IAM handwriting recognition, centered on `scripts/Better_Segmentation2Keras_v5.py`.
See [the file guide](docs/FILE_GUIDE.md) for every file and its role.

## Run Better Segmentation v5

```powershell
python scripts/Better_Segmentation2Keras_v5.py
```

It requires `models/pred.keras`, `data/forms_data.csv`, and IAM form PNGs.
Set the script's `host` variable to your IAM forms directory, keeping the trailing
slash. The images are not included. Required third-party imports include NumPy,
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
