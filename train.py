"""
Training Script — Handwritten Text Recognition using TrOCR
Fine-tuned on IAM Words Dataset (2,500 samples) using Google Colab GPU.
"""

# STEP 1 — Install dependencies
# !pip install -q transformers datasets torch torchvision pillow accelerate

# STEP 2 — Load base model from HuggingFace
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

MODEL_CHECKPOINT = "microsoft/trocr-base-handwritten"
processor = TrOCRProcessor.from_pretrained(MODEL_CHECKPOINT)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_CHECKPOINT)

# STEP 3 — Mount Google Drive (IAM Words dataset stored here)
# from google.colab import drive
# drive.mount("/content/drive")

import os
import pandas as pd

BASE_PATH = "/content/drive/MyDrive/iam words/iam_words"
WORDS_TXT = os.path.join(BASE_PATH, "words.txt")
WORDS_DIR = os.path.join(BASE_PATH, "words")

# STEP 4 — Parse words.txt to get image paths and labels
data = []
with open(WORDS_TXT, "r") as f:
    for line in f:
        if line.startswith("#"):
            continue
        parts = line.strip().split()
        if len(parts) < 9:
            continue
        word_id = parts[0]
        status = parts[1]
        if status != "ok":
            continue
        text = " ".join(parts[8:])
        folder1 = word_id.split("-")[0]
        folder2 = "-".join(word_id.split("-")[:2])
        image_path = f"{WORDS_DIR}/{folder1}/{folder2}/{word_id}.png"
        if os.path.exists(image_path):
            data.append({"image_path": image_path, "text": text})

df = pd.DataFrame(data)
print("Total IAM samples:", len(df))

# STEP 5 — Limit to 2500 samples to avoid Colab OOM crash
df = df.sample(n=2500, random_state=42)
print("Using samples:", len(df))

# STEP 6 — Create HuggingFace Dataset
from datasets import Dataset
from PIL import Image
import numpy as np

dataset = Dataset.from_pandas(df)

# STEP 7 — Filter corrupt images
def is_valid_image(example):
    try:
        img = Image.open(example["image_path"])
        img.verify()
        return True
    except:
        return False

dataset = dataset.filter(is_valid_image)
print("Valid images:", len(dataset))

# STEP 8 — Preprocess
# KEY FIX: IAM images are grayscale — must convert to RGB for TrOCR
def preprocess(example):
    image = Image.open(example["image_path"])
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = np.array(image)

    pixel_values = processor(
        image, return_tensors="pt"
    ).pixel_values[0]

    labels = processor.tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=64
    ).input_ids

    return {"pixel_values": pixel_values, "labels": labels}

processed_dataset = dataset.map(
    preprocess,
    remove_columns=dataset.column_names
)

# STEP 9 — Configure decoder tokens
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.eos_token_id = processor.tokenizer.sep_token_id

# STEP 10 — Training arguments (GPU)
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./trocr_iam_words",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    num_train_epochs=5,
    learning_rate=2e-5,
    fp16=True,
    logging_steps=100,
    save_steps=1000,
    save_total_limit=2,
    dataloader_num_workers=2,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset,
    tokenizer=processor
)

trainer.train()

# STEP 11 — Save model locally
model.save_pretrained("handwritten_trocr_model")
processor.save_pretrained("handwritten_trocr_model")

# STEP 12 — Zip and download from Colab
# !zip -r handwritten_trocr_model.zip handwritten_trocr_model
# from google.colab import files
# files.download("handwritten_trocr_model.zip")

# STEP 13 — Upload to HuggingFace Hub
# huggingface-cli login
# huggingface-cli upload yash2907/handwritten-trocr-iam handwritten_trocr_model --repo-type model
