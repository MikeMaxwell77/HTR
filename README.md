# HTR

Handwritten text recognition experiments using the IAM dataset, Keras models,
word segmentation, and evaluation of predictions refined with ChatGPT.

## File organization

| Folder | Contents |
| --- | --- |
| `scripts/` | Recognition, segmentation, training, data preparation, and metrics scripts |
| `scripts/archive/` | Preserved duplicate versions of the word-list and forms-CSV generators |
| `data/` | `forms_data.csv` and `word_dataset_info.csv` |
| `models/` | Saved Keras model (`pred.keras`) and HTR weights |
| `outputs/` | Prediction results and the detected-word visualization |
| `docs/` | Demo video link |

## Main workflow

Start with `scripts/Better_Segmentation2Keras_v5.py`. It loads
`models/pred.keras` and `data/forms_data.csv`, segments forms, and writes
`outputs/model_predictions.csv`. The existing ChatGPT-refined predictions are
in `outputs/model_predictions_chatGPT_responses.csv`; evaluate them with
`scripts/Metrics Maker2.py`.

Run scripts from the repository root, quoting filenames that contain spaces:

```powershell
python scripts/Better_Segmentation2Keras_v5.py
python "scripts/Metrics Maker2.py"
```

Active scripts resolve relocated CSVs, models, and generated results relative
to the repository. Original script names are preserved.

## Dataset and environment requirements

The IAM images and annotation text files are not included. Some experiments
still use the original author's absolute dataset paths, an `os.chdir` call,
or root-relative `words/` and `words.txt` paths. Configure those for your local
IAM installation before running. `human_text_recognition.py` expects IAM
`words.txt` and `words/` under `data/` and includes notebook shell commands
such as `!wget`; run it in a compatible notebook environment or adapt those
commands first. Install each script's imported dependencies as needed.

Archived scripts retain their original contents and paths for reference.
