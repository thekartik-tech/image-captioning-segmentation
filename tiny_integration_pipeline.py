import os
from tiny_inference_captioning import TinyCaptionGenerator
from tiny_inference_segmentation import TinySegmenter

caption_model = TinyCaptionGenerator("models/encoder.pth", "models/decoder.pth", "vocab.pkl")

segmentation_model = TinySegmenter("models/tiny_unet.pth")

def full_inference(image_path):
    caption = caption_model.predict(image_path)
    mask = segmentation_model.predict(image_path)
    return caption, mask

