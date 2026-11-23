"""Structured visual description (colour, pattern, material...) of one item crop."""
import json

import cv2

COLORS = [
    "black", "white", "grey", "silver", "beige", "cream", "brown", "tan", "red", "maroon",
    "pink", "orange", "yellow", "gold", "olive", "green", "teal", "blue", "navy", "light blue",
    "purple", "lavender", "multicolor", "transparent",
]

PATTERNS = [
    "solid", "striped", "checked", "plaid", "floral", "polka dot", "printed", "graphic",
    "animal print", "camouflage", "textured", "embroidered", "other",
]

SCHEMA = {
    "type": "object",
    "properties": {
        "item": {"type": "string"},
        "primary_color": {"type": "string", "enum": COLORS},
        "secondary_colors": {"type": "array", "items": {"type": "string", "enum": COLORS}},
        "pattern": {"type": "string", "enum": PATTERNS},
        "material": {"type": "string"},
        "style": {"type": "string"},
        "brand_or_text": {"type": "string"},
        "notable_details": {"type": "array", "items": {"type": "string"}},
        "summary": {"type": "string"},
    },
    "required": ["item", "primary_color", "secondary_colors", "pattern", "material", "style",
                 "brand_or_text", "notable_details", "summary"],
}

PROMPT = """This image is a crop from a video frame. A detector labelled the main item "{label}" ({kind}).
Describe only that main item. Ignore the background, other objects and the person's skin, hair and face.

Return JSON with:
- item: what the item is, in 1-3 words
- primary_color: the dominant colour of the item, from the allowed list
- secondary_colors: other clearly visible colours of the item (empty list if none)
- pattern: from the allowed list
- material: likely material, e.g. cotton, denim, leather, metal, glass, plastic, wood; "unknown" if unclear
- style: short style or type, e.g. "long-sleeve casual shirt", "aviator sunglasses"
- brand_or_text: any readable brand, logo or text on the item; "" if none
- notable_details: up to 4 short visible details useful for shopping (e.g. "collar", "zip front", "round dial")
- summary: one short shopping-style sentence

Allowed colours: {colors}
Allowed patterns: {patterns}
Only describe what is visible. Do not guess brands."""

MIN_SIDE = 112


class ItemDescriber:

    def __init__(self, client, model_name, keep_alive="10m"):
        self.client = client
        self.model_name = model_name
        self.keep_alive = keep_alive

    @staticmethod
    def _encode(crop_bgr):
        h, w = crop_bgr.shape[:2]
        if min(h, w) < MIN_SIDE:
            # Very small video crops are upscaled so the model sees enough detail.
            scale = MIN_SIDE / min(h, w)
            crop_bgr = cv2.resize(crop_bgr, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_CUBIC)
        ok, encoded = cv2.imencode(".png", crop_bgr)
        if not ok:
            raise ValueError("Cannot encode crop")
        return encoded.tobytes()

    def describe(self, crop_bgr, label, kind):
        """Return a description dict; failures are reported in the dict, not raised."""
        if crop_bgr is None or crop_bgr.size == 0:
            return dict(error="empty crop", model=self.model_name)
        prompt = PROMPT.format(label=label, kind=kind, colors=", ".join(COLORS), patterns=", ".join(PATTERNS))
        try:
            response = self.client.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt, "images": [self._encode(crop_bgr)]}],
                format=SCHEMA,
                options={"temperature": 0, "num_predict": 400},
                keep_alive=self.keep_alive,
            )
            content = response["message"]["content"] or ""
            if not content.strip():
                reason = ("model used all tokens on thinking; use a non-thinking model such as "
                          "qwen3-vl:4b-instruct" if response["message"].get("thinking") else "model returned no text")
                return dict(error=f"Empty response: {reason}", model=self.model_name)
            data = json.loads(content)
        except Exception as error:
            return dict(error=f"{type(error).__name__}: {error}", model=self.model_name)
        primary = str(data.get("primary_color", "")).lower().strip()
        secondary = []
        for color in data.get("secondary_colors", []):
            color = str(color).lower().strip()
            if color in COLORS and color != primary and color not in secondary:
                secondary.append(color)
        data["primary_color"] = primary
        data["secondary_colors"] = secondary
        data["notable_details"] = [str(d) for d in data.get("notable_details", [])][:4]
        data["model"] = self.model_name
        return data

    def release(self):
        """Unload the model from Ollama's GPU memory."""
        try:
            self.client.generate(model=self.model_name, prompt="", keep_alive=0)
        except Exception:
            pass
