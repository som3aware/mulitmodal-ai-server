import torch
from PIL import Image
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
)


IMAGE_DESCRIPTION_PROMPT = "Describe the image in detail"

device = "cuda" if torch.cuda.is_available() else "cpu"


# BLIP processor
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
blip_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-large"
)

# run on GPU for faster responses
blip_model.to(device)


def get_image_caption(image: Image):
    inputs = blip_processor(image, return_tensors="pt")
    out = blip_model.generate(**inputs)
    return blip_processor.decode(out[0], skip_special_tokens=True)
