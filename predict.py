"""
Track 1 – predict.py
=====================
Implement load_model() and predict() only.
DO NOT modify anything below the marked line.

Self-evaluate on val set:
    INPUT_CSV  = "val.csv"
    IMAGE_DIR  = "val/images/"

Final submission (paths must be set to test before submitting):
    INPUT_CSV  = "test.csv"
    IMAGE_DIR  = "test/images/"
"""

import os
import pandas as pd
import timm
from PIL import Image
from mpmath.identification import transforms
from sympy.printing.pytorch import torch
from torchvision import transforms as T

# ==============================================================================
# CHANGE THESE PATHS IF NEEDED
# ==============================================================================

INPUT_CSV   = "val.csv"
IMAGE_DIR   = "val/images/"
OUTPUT_PATH = "predictions.csv"
MODEL_PATH  = "model/"

# ==============================================================================
# YOUR CODE — IMPLEMENT THESE TWO FUNCTIONS
# ==============================================================================

def load_model():
    device = torch.device("cpu")

    model = timm.create_model('resnet50', pretrained=False, num_classes=102)
    model.load_state_dict(torch.load("model/model9.pth"))
    model.to(device)
    model.eval()
    return model

    # set images to be similar to training images


predict_tfms = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def predict(model, images: list) -> list[int]:
    preds = []
    device = torch.device("cpu")
    for image in images:
        x = predict_tfms(image).unsqueeze(0)
        x = x.to(device)

        with torch.no_grad():
            output = model(x)
            predicted = torch.argmax(output).item()

        preds.append(predicted+1)
    return preds



    raise NotImplementedError("Implement predict()")

# ==============================================================================
# DO NOT MODIFY ANYTHING BELOW THIS LINE
# ==============================================================================

def _load_images(df):
    images, missing = [], []
    for _, row in df.iterrows():
        path = os.path.join(IMAGE_DIR, row["filename"])
        if os.path.exists(path):
            images.append(Image.open(path).convert("RGB"))
        else:
            missing.append(row["filename"])
            images.append(None)
    if missing:
        print(f"WARNING: {len(missing)} image(s) not found. First few: {missing[:5]}")
    return images

def main():
    df = pd.read_csv(INPUT_CSV, dtype=str)
    missing_cols = {"image_id", "filename"} - set(df.columns)
    if missing_cols:
        raise ValueError(f"Input CSV missing columns: {missing_cols}")
    print(f"Loaded {len(df):,} images from {INPUT_CSV}")

    images = _load_images(df)
    model  = load_model()
    preds  = predict(model, images)

    if len(preds) != len(df):
        raise ValueError(f"predict() returned {len(preds)} predictions for {len(df)} images.")

    out = df[["image_id"]].copy()
    out["label"] = [int(p) for p in preds]
    out.to_csv(OUTPUT_PATH, index=False)
    print(f"Predictions saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()