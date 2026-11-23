import os
import json
import re
import gc
from time import perf_counter
import torch
from PIL import Image
from transformers import Florence2ForConditionalGeneration, AutoProcessor
from ollama import Client
from dotenv import load_dotenv

load_dotenv()


class PromptGenerator:

    # --------------------------------------
    # INIT
    # --------------------------------------
    def __init__(self, clothing_dir, object_dir, device="cpu", batch_size=1):

        self.clothing_dir = clothing_dir
        self.object_dir = object_dir
        self.device = str(device)
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        self.batch_size = batch_size
        self.florence_model = None
        self.processor = None
        self.ollama_client = Client(
            host=os.getenv(
                "OLLAMA_HOST",
                "http://127.0.0.1:11434"
            ),
            timeout=180,
        )

        # --------------------------------------
        # LOAD FLORENCE
        # --------------------------------------

        self.florence_prompt = "<OD>"

        
        # # --------------------------------------
        # # GEMINI CLIENT
        # # --------------------------------------

        # print("Initializing Gemini client...")

        # if not os.getenv("GEMINI_API_KEY"):
        #     raise EnvironmentError(
        #         "GEMINI_API_KEY environment variable not set."
        #     )

        # self.client = genai.Client(
        #     api_key=os.getenv("GEMINI_API_KEY")
        # )

        # --------------------------------------
        # ALLOWED CLOTHING ITEMS
        # --------------------------------------

        self.CLOTHING_ITEMS = [
            "shirt",
            "t-shirt",
            "tshirt",
            "hoodie",
            "sweater",
            "jacket",
            "jeans",
            "pants",
            "trousers",
            "shorts",
            "skirt",
            "miniskirt",
            "dress",
            "gown",
            "jumpsuit",
            "kurta",
            "saree",
            "lehenga",
            "salwar"
        ]

    # --------------------------------------
    # LOAD IMAGES
    # --------------------------------------
    def image_paths(self, folder):
        return sorted(
            os.path.join(directory, f)
            for directory, _, files in os.walk(folder)
            for f in files
            if f.lower().endswith(
                (".jpg", ".jpeg", ".png", ".webp")
            )
        )

    def load_images(self, folder):
        return self._open_images(self.image_paths(folder))

    def _open_images(self, image_paths):

        images = []

        try:
            for path in image_paths:
                with Image.open(path) as source:
                    images.append(source.convert("RGB"))
        except Exception:
            for image in images:
                image.close()
            raise

        return images

    def _load_florence(self):
        if self.florence_model is not None:
            return
        started = perf_counter()
        print(f"Loading Florence model on {self.device}...", flush=True)
        self.florence_model = Florence2ForConditionalGeneration.from_pretrained(
            "florence-community/Florence-2-large",
            dtype=torch.float16 if self.device.startswith("cuda") else torch.float32,
        ).to(self.device)
        self.florence_model.eval()
        self.processor = AutoProcessor.from_pretrained(
            "florence-community/Florence-2-large"
        )
        print(f"Florence loaded in {perf_counter() - started:.1f}s", flush=True)

    def release_florence(self):
        self.florence_model = None
        self.processor = None
        gc.collect()
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()

    def extract_folder_tags(self, folder):
        paths = self.image_paths(folder)
        tags = []
        print(f"\nProcessing {len(paths)} images from {folder}", flush=True)
        for start in range(0, len(paths), self.batch_size):
            batch = paths[start:start + self.batch_size]
            print(f"Images {start + 1}-{start + len(batch)}/{len(paths)}...", flush=True)
            started = perf_counter()
            images = self._open_images(batch)
            try:
                tags.extend(self.extract_tags(images))
            finally:
                for image in images:
                    image.close()
            print(f"Batch completed in {perf_counter() - started:.1f}s", flush=True)
        return sorted(set(tags))

    # --------------------------------------
    # FLORENCE TAG EXTRACTION
    # --------------------------------------
    def extract_tags(self, images):

        if len(images) == 0:
            return []

        self._load_florence()

        inputs = self.processor(
            text=[self.florence_prompt] * len(images),
            images=images,
            return_tensors="pt",
            padding=True
        ).to(self.device)

        # Keep token IDs as integers; only the image tensor needs model precision.
        inputs["pixel_values"] = inputs["pixel_values"].to(self.florence_model.dtype)

        with torch.no_grad():

            generated_ids = (
                self.florence_model.generate(
                    **inputs,
                    max_new_tokens=20
                )
            )

        captions = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        print("\nFlorence Raw Output:")

        for c in captions:
            print(c)

        cleaned_tags = []

        for caption in captions:

            caption = caption.lower()
            caption = caption.replace(".", "")

            parts = caption.split(",")

            for part in parts:

                tag = part.strip()

                if tag:
                    cleaned_tags.append(tag)

        cleaned_tags = list(set(cleaned_tags))

        print("\nCleaned Tags:")
        print(cleaned_tags)

        return cleaned_tags

    # --------------------------------------
    # REMOVE NON PRODUCTS
    # --------------------------------------
    def remove_non_products(self, tags):

        NON_PRODUCT_TERMS = {
            "man",
            "woman",
            "person",
            "people",
            "girl",
            "boy",
            "human",
            "model",
            "face",
            "body",
            "adult",
            "child",
            "background",
            "scene"
        }

        filtered=[]

        for t in tags:

            t=t.strip().lower()

            if t not in NON_PRODUCT_TERMS:
                filtered.append(t)

        return filtered

    # --------------------------------------
    # SHORTEN INPUT
    # protects prompt size
    # --------------------------------------
    def compress_tags(self, tags):

        tags = sorted(set(tags))

        cleaned=[]

        for t in tags:

            t=t.strip()

            if len(t)>40:
                t=t[:40]

            cleaned.append(t)

        # keep only first 50 tags
        return cleaned[:50]


    # --------------------------------------
    # GEMINI LLM PROCESSING
    # --------------------------------------
    def process_clothing_tags_with_llm(self, tags):

        if len(tags) == 0:
            return []

        tags = self.compress_tags(tags)

        allowed_string = ", ".join(
            self.CLOTHING_ITEMS
        )

        tag_string = ", ".join(tags)

        # small prompt
        prompt = f"""
        Convert noisy Florence-generated fashion labels into valid clothing categories.

        Allowed Categories:
        {allowed_string}

        Input Labels:
        {tag_string}

        Rules:
        1. return only clothing categories supported by the input labels
        2. remove duplicates
        3. fix merged words
        4. if no clothing category is supported, return {{"clothing": []}}
        5. never add dress or any other category as a default

        Return ONLY valid JSON.

        Example:

        {{
            "clothing": ["dress", "jacket"]
        }}
        """

        print("Normalizing clothing labels with Ollama...", flush=True)
        started = perf_counter()
        response = self.ollama_client.chat(
            model="qwen2.5:3b-instruct",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a fashion detection assistant. "
                        "Always return valid JSON only."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            format="json",
            options={
                "temperature": 0,
                "num_predict": 256,
            },
            keep_alive=0,
        )
        print(f"Ollama completed in {perf_counter() - started:.1f}s", flush=True)

        result = response["message"]["content"]

        print("\nRAW OLLAMA RESPONSE:")
        print(result)

        result = (
            result
            .replace("```json", "")
            .replace("```", "")
            .strip()
        )

        match = re.search(
            r"\{.*\}",
            result,
            re.DOTALL
        )

        if not match:
            raise ValueError(
                "No JSON found in Ollama response"
            )

        json_text = match.group()

        print("\nEXTRACTED JSON:")
        print(json_text)

        data = json.loads(json_text)

        cleaned = list(set([
            t.lower().strip()
            for t in data.get(
                "clothing",
                []
            )
        ]))

        print("\nFinal Clothing Tags:")
        print(cleaned)

        return cleaned


    # --------------------------------------
    # MAIN PIPELINE
    # --------------------------------------
    def generate_prompt(self):

        try:
            clothing_tags = self.extract_folder_tags(self.clothing_dir)
            object_tags = self.extract_folder_tags(self.object_dir)
        finally:
            # Florence and Ollama should not compete for the same small GPU.
            self.release_florence()

        clothing_tags=(
            self.remove_non_products(
                clothing_tags
            )
        )

        object_tags=(
            self.remove_non_products(
                object_tags
            )
        )

        clothing_tags=(
            self.process_clothing_tags_with_llm(
                clothing_tags
            )
        )

        print("\nFinal Clothing Tags:")
        print(clothing_tags)

        print("\nFinal Object Tags:")
        print(object_tags)

        clothing_prompt=" . ".join(
            sorted(clothing_tags)
        )

        object_prompt=" . ".join(
            sorted(object_tags)
        )

        print("\nClothing Prompt:")
        print(clothing_prompt)

        print("\nObject Prompt:")
        print(object_prompt)

        return {
            "clothing_prompt":
                clothing_prompt,

            "object_prompt":
                object_prompt
        }


# Example
# generator = PromptGenerator(
#     clothing_dir="clothes",
#     object_dir="objects",
#     device="cuda"
# )

# result=generator.generate_prompt()

# print(result)
