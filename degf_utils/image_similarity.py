import torch
from PIL import Image
from transformers import AutoProcessor, CLIPModel
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
clip_model_path = "/home/ciram25-liurp/models/clip-vit-base-patch32"
processor = AutoProcessor.from_pretrained(clip_model_path)
model = CLIPModel.from_pretrained(clip_model_path).to(device)

def get_clip_similarity(image1, image2):
    #Extract features from image1
    with torch.no_grad():
        inputs1 = processor(images=image1, return_tensors="pt").to(device)
        image_features1 = model.get_image_features(**inputs1)

    #Extract features from image2
    with torch.no_grad():
        inputs2 = processor(images=image2, return_tensors="pt").to(device)
        image_features2 = model.get_image_features(**inputs2)

    #Compute their cosine similarity and convert it into a score between 0 and 1
    cos = nn.CosineSimilarity(dim=0)
    sim = cos(image_features1[0],image_features2[0]).item()

    return sim