# Track 1 – Image Classification

## Task
Fine-grained image classification across **102 categories**. Your model must
predict the correct class label for each image in the withheld test set.
Organizers will run your `predict.py` against the test set after submission.

---

## Files Provided

| File / Folder | Description |
|---------------|-------------|
| `train/images/` | Training images in a flat folder |
| `val/images/` | Validation images in a flat folder (clean labels) |
| `train.csv` | `image_id`, `filename`, `label` for training set |
| `val.csv` | `image_id`, `filename`, `label` for validation set |
| `label_list.txt` | All valid class labels (integers 1–102) |
| `predict.py` | Inference template — implement this and submit it |

---

## Categories

**Category 1 – With Pretrained Models**
Pretrained vision model weights are allowed.
Fine-tuning pretrained weights is permitted. No large language models.

**Category 2 – Without Pretrained Models**
All model parameters must be trained from scratch using only the provided
training data. No pretrained weights or external feature extractors.

---

## Label Format
- Labels are integers from **1 to 102**
- One label per image — single-label classification
- Use only labels found in `label_list.txt`

---

## How to Self-Evaluate
Use `val.csv` and `val/images/` to evaluate locally. Set the paths at the
top of `predict.py`:

```python
INPUT_CSV   = "val.csv"
IMAGE_DIR   = "val/images/"
OUTPUT_PATH = "val_predictions.csv"
MODEL_PATH  = "model/"
```

Then run:
```bash
python predict.py
```

Compute your own metrics from `val_predictions.csv` against `val.csv`.
Your final leaderboard ranking is based on **Macro F1** on the hidden test
set evaluated by organizers.

---

## What to Submit
A single zip file named `teamname_track1_cat[1or2].zip`:

```
teamname_track1_cat[1or2].zip
├── predict.py        ← paths must be set to test before submitting
├── train.py          ← your training script
├── model/            ← saved model weights
├── requirements.txt  ← all dependencies with pinned versions
└── report.pdf        ← technical description (max 2 pages)
```

### `predict.py` Requirements
Before submitting, ensure paths at the top of `predict.py` are:

```python
INPUT_CSV   = "test.csv"
IMAGE_DIR   = "test/images/"
OUTPUT_PATH = "predictions.csv"
MODEL_PATH  = "model/"
```

Organizers will place your `predict.py` and `model/` alongside the withheld
test set and run:
```bash
pip install -r requirements.txt
python predict.py
```

**Your `predict.py` must:**
- Load weights from `MODEL_PATH`
- Accept PIL images as input inside `predict()`
- Return one integer label per image
- Run without internet access
- Complete in under 10 minutes
- Not retrain the model — inference only

### `report.pdf` (max 2 pages)
- Preprocessing and augmentation steps
- Model architecture
- Training procedure and hyperparameters
- Explicit statement of category compliance (Category 1 or 2)

---

## Evaluation Metric
Submissions are ranked by **Macro F1 score** on the hidden test set.

Macro F1 computes F1 independently for each of the 102 classes and averages
them equally — a model that ignores minority classes will be penalized heavily.

---

## Rules
- No large language models in any form
- Only the provided dataset files may be used — no external image data
- **Category 1:** pretrained vision model weights allowed
- **Category 2:** all weights must be trained from scratch on provided data only
- Submissions must be fully reproducible
- Non-reproducible submissions will be disqualified