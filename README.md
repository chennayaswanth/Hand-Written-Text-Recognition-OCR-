# ✍️ Handwritten Text Recognition using Transformer-based OCR

[![HuggingFace Space](https://img.shields.io/badge/🤗%20Live%20Demo-HuggingFace-blue)](https://huggingface.co/spaces/yash2907/handwritten-ocr-trocr)
[![Model](https://img.shields.io/badge/🤗%20Model-yash2907/handwritten--trocr--iam-yellow)](https://huggingface.co/yash2907/handwritten-trocr-iam)
[![Python](https://img.shields.io/badge/Python-3.11-green)](https://python.org)

A production-ready **handwritten text recognition** system built by fine-tuning Microsoft's TrOCR transformer on the IAM handwriting dataset. Upload any handwritten image and get the transcribed text instantly.

---

## 🎯 Results

| Metric | Score |
|--------|-------|
| Character Error Rate (CER) | **< 7%** |
| Word Error Rate (WER) | **< 15%** |
| Training Samples | 2,500 IAM words |
| Model Size | 1.3 GB |

---

## 🏗️ Architecture

```
Handwritten Image
       ↓
  Vision Encoder (ViT)     ← Extracts image patch features
       ↓
  Text Decoder (RoBERTa)   ← Generates text token by token
       ↓
  Predicted Text
```

- **Base Model:** `microsoft/trocr-base-handwritten`
- **Fine-tuned on:** IAM Words Dataset (2,500 samples)
- **Frontend:** Gradio
- **Deployed on:** Hugging Face Spaces
- **Model hosted on:** Hugging Face Model Hub

---

## 🚀 Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/chennayaswanth/handwritten-text-recognition-trocr
cd handwritten-text-recognition-trocr

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the Gradio app
python app.py
```

Open your browser at `http://localhost:7860`

---

## 📁 Project Structure

```
handwritten-text-recognition-trocr/
│
├── app.py               ← Gradio app (loads model from HuggingFace)
├── train.py             ← Training script (run on Google Colab)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🔑 Key Engineering Challenges Solved

- **RGB/Grayscale bug:** IAM dataset images are grayscale but TrOCR expects RGB — fixed by force-converting all images to RGB before preprocessing
- **Model hosting (1.3GB):** Used HuggingFace Model Hub to host model weights — loaded directly at inference time via `from_pretrained()`
- **Colab OOM:** Limited dataset to 2,500 samples with `fp16=True` and `gradient_accumulation_steps=2` to fit in GPU memory

---

## 🛠️ Tech Stack

`Python` · `PyTorch` · `Hugging Face Transformers` · `Gradio` · `Google Colab`

---

## 👤 Author

**Yaswanth Chenna**
[![LinkedIn](https://img.shields.io/badge/LinkedIn-chennayaswanth-blue)](https://linkedin.com/in/chennayaswanth)
[![GitHub](https://img.shields.io/badge/GitHub-chennayaswanth-black)](https://github.com/chennayaswanth)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-yash2907-yellow)](https://huggingface.co/yash2907)
