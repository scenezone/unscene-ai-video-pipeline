import sys
from pathlib import Path
import json
import argparse
import hashlib
import os
from time import perf_counter

# Allow imports from project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))



# ----------------------------
# CONFIG
# ----------------------------

CLOTHING_DIR = PROJECT_ROOT / "data/productimages/clothing"
OBJECT_DIR = PROJECT_ROOT / "data/productimages/objects"

DEVICE = "auto"

SAVE_PROMPT = True
PROMPT_OUTPUT_FILE = PROJECT_ROOT / "config/dino_prompt.json"
CACHE_FILE = PROJECT_ROOT / "outputs/prompt_generation/cache.json"


def input_fingerprint():
    """Invalidate cached results after image or generator code changes."""
    digest = hashlib.sha256()
    for folder in (CLOTHING_DIR, OBJECT_DIR):
        folder = Path(folder)
        if not folder.is_dir():
            raise FileNotFoundError(f"Product image folder does not exist: {folder}")
        digest.update(str(folder.resolve()).encode())
        for path in sorted(folder.rglob("*")):
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                digest.update(str(path.relative_to(folder)).encode())
                image_digest = hashlib.sha256()
                with path.open("rb") as source:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        image_digest.update(chunk)
                digest.update(image_digest.digest())
    for path in (Path(__file__), PROJECT_ROOT / "services/prompt_generation/prompt_generator.py"):
        digest.update(path.read_bytes())
    return digest.hexdigest()


def load_cached_prompt(fingerprint):
    try:
        cached = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        if not isinstance(cached, dict) or cached.get("fingerprint") != fingerprint:
            return None
        data = cached.get("prompt_data")
        if isinstance(data, dict) and all(
            isinstance(data.get(key), str) for key in ("clothing_prompt", "object_prompt")
        ):
            return data
    except (OSError, ValueError):
        pass
    return None


# ----------------------------
# RUN PROMPT GENERATOR
# ----------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate product prompts with bounded memory and caching.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=DEVICE)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--cpu-threads", type=int, default=max(1, min(4, (os.cpu_count() or 2) // 2)))
    parser.add_argument("--refresh", action="store_true", help="Regenerate even if images are unchanged")
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.cpu_threads < 1:
        parser.error("--batch-size and --cpu-threads must be positive")
    started = perf_counter()
    fingerprint = input_fingerprint()
    prompt_data = None if args.refresh else load_cached_prompt(fingerprint)
    if prompt_data is not None:
        print("Using cached prompts (images unchanged). Use --refresh to regenerate.", flush=True)
    else:
        # Cache hits avoid importing or initializing either model stack.
        import torch
        from services.prompt_generation.prompt_generator import PromptGenerator

        device = args.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            parser.error("CUDA is unavailable in this Python environment. Use --device cpu.")
        if device == "cpu":
            torch.set_num_threads(args.cpu_threads)
        print(f"\nGenerating prompts on {device}, batch size {args.batch_size}...", flush=True)
        generator = PromptGenerator(
            clothing_dir=CLOTHING_DIR,
            object_dir=OBJECT_DIR,
            device=device,
            batch_size=args.batch_size,
        )
        try:
            prompt_data = generator.generate_prompt()
        except torch.cuda.OutOfMemoryError:
            raise SystemExit(
                "GPU memory exhausted. Close other GPU workloads or rerun with --device cpu."
            ) from None
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        CACHE_FILE.write_text(json.dumps({"fingerprint": fingerprint, "prompt_data": prompt_data}), encoding="utf-8")

    print("\n==============================")
    print("Generated Prompt Data")
    print("==============================")

    print(json.dumps(prompt_data, indent=4))

    # ----------------------------
    # OPTIONAL SAVE
    # ----------------------------

    if SAVE_PROMPT:

        PROMPT_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

        with open(PROMPT_OUTPUT_FILE, "w") as f:
            json.dump(prompt_data, f, indent=4)

        print("\nPrompt saved to:", PROMPT_OUTPUT_FILE)
    print(f"Total time: {perf_counter() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
