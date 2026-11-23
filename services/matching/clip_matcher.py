import cv2
import torch
import numpy as np
from PIL import Image
from pathlib import Path


class CLIPMatcher:

    def __init__(self, model, preprocess, device="cpu"):
        self.model = model
        self.preprocess = preprocess
        self.device = device

        self.product_embeddings = {}
        self.product_names = []

    # ------------------------
    # IMAGE EMBEDDING
    # ------------------------

    def get_embedding(self, image_np):

        image = Image.fromarray(image_np)

        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():

            embedding = self.model.encode_image(image_tensor)

            embedding /= embedding.norm(dim=-1, keepdim=True)

        return embedding

    # ------------------------
    # LOAD PRODUCT CATALOG
    # ------------------------

    def load_product_catalog(self, product_folder):

        product_folder = Path(product_folder)

        print("Loading product embeddings...")

        for img_path in product_folder.iterdir():

            if img_path.suffix.lower() not in [".jpg",".jpeg",".png",".webp"]:
                continue

            img = cv2.imread(str(img_path))

            if img is None:
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            emb = self.get_embedding(img_rgb)

            self.product_embeddings[img_path.name] = emb

        self.product_names = list(self.product_embeddings.keys())

        print(f"Loaded {len(self.product_embeddings)} products")

    # ------------------------
    # MATCH SEGMENTED OBJECT
    # ------------------------

    def match(self, crop_np, threshold=0.50):

        crop_embedding = self.get_embedding(crop_np)

        best_score = -1
        best_product = None

        for name, ref_emb in self.product_embeddings.items():

            similarity = (crop_embedding @ ref_emb.T).item()

            if similarity > best_score:
                best_score = similarity
                best_product = name

        matched = best_score >= threshold

        return matched, best_product, best_score