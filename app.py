import gradio as gr
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# -------------------------------------------------------
# Load model from HuggingFace Hub (no local weights needed)
# -------------------------------------------------------
MODEL_NAME = "yash2907/handwritten-trocr-iam"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = TrOCRProcessor.from_pretrained(MODEL_NAME, use_fast=False)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
model.to(device)
model.eval()


def predict(image):
    """
    Takes a PIL Image input from Gradio and returns predicted text.
    """
    if image is None:
        return "Please upload an image."

    if image.mode != "RGB":
        image = image.convert("RGB")

    pixel_values = processor(
        image, return_tensors="pt"
    ).pixel_values.to(device)

    with torch.no_grad():
        generated_ids = model.generate(pixel_values)

    predicted_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True
    )[0]

    return predicted_text.strip()


# -------------------------------------------------------
# Gradio Interface
# -------------------------------------------------------
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Handwritten Image"),
    outputs=gr.Textbox(label="Predicted Text", lines=3),
    title="✍️ Handwritten Text Recognition",
    description=(
        "Upload an image of handwritten text and the model will transcribe it.\n\n"
        "Fine-tuned **Microsoft TrOCR** on 2,500 IAM handwriting samples.\n"
        "**Character Error Rate (CER) < 7% | Word Error Rate (WER) < 15%**"
    ),
    theme=gr.themes.Soft(),
)

if __name__ == "__main__":
    demo.launch()
