# File guide

## What Better Segmentation v5 needs

`Better_Segmentation2Keras_v5.py` is the CLI entry point for the modules in
`scripts/segmentation/`. These modules are required alongside the entry point.
No data-preparation, training, example, or evaluation script needs to run first. Running inference requires:

1. `models/pred.keras`: the pretrained recognition model.
2. `data/forms_data.csv`: form IDs and reference text (`Form ID`, `Form Data`).
3. External IAM form images named `<form_id>.png`, found through the script's
   `--forms-dir` argument (default in `segmentation/config.py`). These images are not in this repository.
4. NumPy, OpenCV (`cv2`), Keras, TensorFlow, Matplotlib, pandas, and TextBlob.

`Make formsCSV.py` is only needed if rebuilding the reference CSV. No training,
RNN, image-loading example, OpenAI example, or metrics script is required first.
The saved model's provenance cannot be established from the filenames alone.

## Every file

| File | Purpose | Needed by v5? |
| --- | --- | --- |
| `scripts/Better_Segmentation2Keras_v5.py` | Detects and groups handwritten regions, crops/preprocesses words, runs the saved model, decodes text, prints TextBlob corrections and overlap scores, displays plots, and exports predictions. | Main entry point; implementation delegated to `segmentation/` |
| `scripts/data_preparation/Make formsCSV.py` | Reads external IAM `ascii/lines.txt`, combines valid lines by form, keeps IDs starting with `j`, and writes `data/forms_data.csv`. Input path is still machine-specific. | Only to regenerate the CSV |
| `scripts/data_preparation/Make word_location_list.py` | Reads external `words.txt`; writes image paths and word labels into `data/word_dataset_info.csv`. | No |
| `scripts/training/CTC_Word.py` | Trains a CNN with bidirectional LSTM and CTC loss; evaluates predictions, plots loss, and saves `models/word_recognition_model.keras`. Has a machine-specific working directory. | No |
| `scripts/training/human_text_recognition.py` | Colab-derived IAM recognition training example with convolutional/recurrent layers and CTC; saves `models/word_recognition_model_alpha.keras`. Contains `!` notebook shell commands and expects `data/words.txt` plus `data/words/`. | No |
| `scripts/evaluation/Metrics Maker2.py` | Compares the existing CSV's `Chat GPT` column with `text_true`; prints cosine similarity and word-overlap scores and averages. Does not directly evaluate v5's new CSV, which lacks `Chat GPT`. | No |
| `scripts/examples/load images and preprocess them.py` | Loads the word metadata CSV, opens one image from root-relative `words/`, resizes it, and shows it. | No |
| `scripts/examples/import openai.py` | Standalone text-cleaning API example with a placeholder key and `openai.ChatCompletion.create` call. Not wired into v5. | No |
| `scripts/archive/Add RNN Simple.py` | Earlier four-word CNN/SimpleRNN classifier (`the`, `of`, `to`, `and`); plots training results and saves `models/CNN_RNN_Basic_combo.keras`. Archived as unnecessary for this workflow. | No |
| `scripts/archive/word_location_list.py` | Duplicate word metadata generator, preserved with its original relative paths. | No |
| `scripts/archive/Make word_location_list_original.py` | Despite its name, duplicates the forms-CSV generator; retains original absolute paths. | No |
| `data/forms_data.csv` | Reference form IDs and complete text used by v5 to locate images and compare predictions. | Yes |
| `data/word_dataset_info.csv` | Individual word image paths and labels for training and preprocessing experiments. | No |
| `models/pred.keras` | Saved recognition model directly loaded by v5. | Yes |
| `models/HTR.weights.h5` | Saved model weights; no current script loads this file. Weights alone require matching model architecture. | No |
| `outputs/model_predictions_chatGPT_responses.csv` | Existing predictions with ChatGPT-refined text and reference text, used by the metrics script. | No |
| `outputs/Detected Words Sorted in Red.png` | Existing visual example of detected word regions. | No |
| `docs/share demo link.txt` | Link to the recorded demonstration. | No |
| `README.md` | Main workflow and execution instructions. | Documentation |
| `docs/FILE_GUIDE.md` | This inventory and dependency explanation. | Documentation |

`CNN test.py` was already deleted in the working tree before this reorganization;
that deletion was preserved. It was a separate four-word CNN classifier, not a
v5 dependency.

## Generated files

The main run writes `outputs/model_predictions.csv` with `form_id`, `text_pred`,
and `text_true` (plus the pandas index). `text_pred` contains the raw predicted
sentence, not the printed TextBlob correction. The detector's visualization
method can also write `outputs/detection_visualization.png` when invoked.

The training scripts' saved filenames are different from `pred.keras`. Do not
assume renaming any newly trained model makes it compatible: input shape,
preprocessing, character vocabulary, and decoding must agree with v5.

## Modular implementation

See the [README module table](../README.md#segmentation-modules) for each file
in `scripts/segmentation/`. Keep this entire package with the v5 entry point.
The pipeline skips unreadable images and empty extractions, and creates the
output directory when needed. Core detector, resize, preprocessing, alternate
segmentation, and decoder function bodies are preserved from v5.
