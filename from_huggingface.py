# -*- coding: utf-8 -*-
from transformers import ViltProcessor, ViltForImagesAndTextClassification
import requests
from PIL import Image

image1 = Image.open("/vc_data/users/taoli1/data/ex0_0.jpg")
image2 = Image.open("/vc_data/users/taoli1/data/ex0_1.jpg")
text = "The left image contains twice the number of dogs as the right image."
model_path="dandelin/vilt-b32-finetuned-nlvr2"
processor = ViltProcessor.from_pretrained(model_path)
model = ViltForImagesAndTextClassification.from_pretrained(model_path)

# prepare inputs
encoding = processor([image1, image2], text, return_tensors="pt")

# forward pass
outputs = model(input_ids=encoding.input_ids, pixel_values=encoding.pixel_values.unsqueeze(0))
logits = outputs.logits
idx = logits.argmax(-1).item()
print("Predicted answer:", model.config.id2label[idx])
