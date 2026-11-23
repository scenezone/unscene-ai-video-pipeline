import os

from ollama import Client, ResponseError


class QwenVLLoader:
    """Connects to a Qwen3-VL model served by Ollama.

    The weights run inside the Ollama process (4-bit), not in this Python process,
    so they fit next to GroundingDINO/SAM/CLIP on a 6 GB GPU. load() returns the
    client, like the other loaders return their model.
    """

    def __init__(self, model_name=None, host=None, timeout=300):
        # "-instruct" answers directly; plain "qwen3-vl:4b" is the thinking variant, which
        # spends its tokens reasoning and returns empty JSON.
        self.model_name = model_name or os.getenv("AI_DESCRIBE_MODEL", "qwen3-vl:4b-instruct")
        self.host = host or os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
        self.timeout = timeout
        self.client = None

    def load(self):
        print(f"Loading Qwen-VL model ({self.model_name}) via Ollama at {self.host}...")
        client = Client(host=self.host, timeout=self.timeout)
        try:
            info = client.show(self.model_name)
        except ResponseError as error:
            if error.status_code == 404:
                raise RuntimeError(
                    f"Ollama does not have {self.model_name}. Run: ollama pull {self.model_name}") from None
            raise
        except ConnectionError:
            raise RuntimeError(
                f"Ollama is not reachable at {self.host}. Start the Ollama app or run: ollama serve") from None
        capabilities = getattr(info, "capabilities", None)
        if capabilities and "vision" not in capabilities:
            raise RuntimeError(f"{self.model_name} is not a vision model (capabilities: {capabilities})")
        self.client = client
        return self.client
