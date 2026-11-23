## Shoppable AI Product (Video → Shoppable Timestamps)

The authenticated local API is available with either **`python main.py`** or **`uvicorn main:app`**. On first startup it prints an initial client ID and secret; enter them in Swagger's Authorize dialog to call protected routes. Clients and jobs are stored in SQLite. See [API authentication](docs/API_AUTH.md) and copy [.env.example](.env.example) for settings.

For separate startup instructions, authentication, every endpoint, request/response examples, and complete processing steps for both execution modes, see the [API and end-to-end pipeline guide](docs/PIPELINE_API_END_TO_END.md).

For the project architecture, see the combined [High-Level and Low-Level Design](docs/HLD_LLD.md).

The Unscene pull worker remains available as **`python -m services.unscene_worker`** and uses the Unscene client credentials in `.env`. Its workflow is documented in the worker and deployment sections of [HLD/LLD](docs/HLD_LLD.md). The historical overview below describes the earlier pipeline.

Turn a video into **shoppable moments** by detecting products/clothing in frames, matching detected objects to your product catalog, and exporting a clean **timestamp JSON** you can plug into a UI.

This repo stitches together:

- **GroundingDINO** for text-prompted object detection (what’s in the frame?)
- **SAM (Segment Anything)** for segmentation (isolate the detected object)
- **CLIP** for image-to-image similarity (match segmented crop to a product image)
- **Florence-2 + Gemini** to auto-generate better detection prompts from your catalog images (optional, cached)

---

## What you get

- **Frame-level visual outputs**
  - Annotated clothing frames: `outputs/video/clothing_frames/`
  - Annotated matched product frames: `outputs/video/object_frames/`
- **Machine-readable timestamps**
  - `outputs/video/video_timestamps.json` with merged \([start, end]\) ranges per product / clothing label

Example JSON shape:

```json
{
  "objects": {
    "Product 2": [
      { "start": 10.0, "end": 12.0 },
      { "start": 40.0, "end": 41.0 }
    ]
  },
  "clothing": {
    "shirt": [
      { "start": 5.0, "end": 7.0 }
    ]
  }
}
```

---

## Project structure (high-level)

```text
pipelines/
  video_pipeline.py               # Orchestrates detection → segmentation → matching → JSON
models/
  dino_loader.py                  # Loads GroundingDINO
  sam_loader.py                   # Loads SAM predictor
  clip_loader.py                  # Loads CLIP + preprocess
services/
  detection/
    detector.py                   # GroundingDINO detector wrapper
    schemas.py                    # Detection dataclass
  segmentation/
    segmenter.py                  # SAM-based segmentation wrapper
  matching/
    clip_matcher.py               # CLIP image similarity matching
  prompt_generation/
    prompt_generator.py           # Florence-2 captions + Gemini categorization → prompts
tests/
  test_video_pipeline.py          # End-to-end runner script (pipeline)
  test_detection.py               # Detection-only runner script
  test_caption_generator.py       # Prompt generation runner script
data/
  testdata/test_video.mp4         # Sample video (your repo currently includes one)
  productimages/                  # Your product catalog images (.jpg/.png/.webp)
weights/
  GroundingDINO/                  # GroundingDINO code + config + weights
  sam-weight/                     # SAM weights (.pth)
outputs/
  video/                          # Pipeline outputs (frames + timestamps JSON)
config/
  prompt_cache.json               # Cached Florence+Gemini prompts
```

---

## How it works (pipeline overview)

`pipelines/video_pipeline.py` does the following:

1. **Sample frames** from the input video (default: `target_fps=1`).
2. **Detect clothing and/or objects** using GroundingDINO with a text prompt.
3. For object detections:
   - Run **SAM segmentation** on the detection box
   - Convert the segmented crop to RGB
   - **Match** that crop against your product catalog using CLIP similarity
4. Save annotated frames and collect timestamps.
5. **Merge nearby timestamps** into time ranges and save `video_timestamps.json`.

Prompt generation (optional) in `services/prompt_generation/prompt_generator.py`:

- Uses **Florence-2** (`florence-community/Florence-2-large`) to produce short tags for each product image.
- Sends the tags to **Gemini** to normalize and split into two buckets:
  - `clothing`: wearable items
  - `objects`: accessories / non-wearables
- Stores the result in `config/prompt_cache.json` so you don’t pay/latency-hit every run.

---

## Requirements

### System

- Python 3.9+ recommended (Windows works; this repo is currently set up for Windows-style paths in the runner scripts)
- A working install of PyTorch compatible with your machine (CPU or CUDA)

### Python dependencies (conceptually)

This project uses:

- `opencv-python`
- `numpy`
- `torch`
- `Pillow`
- `transformers`
- `open_clip_torch` (imported as `open_clip`)
- `python-dotenv`
- `google-genai`
- GroundingDINO + SAM codebases (pulled in via your `weights/` layout / environment)

Because dependency versions vary a lot by GPU/torch build, **pinning is left to you**. If you want, I can generate a `requirements.txt`/`pyproject.toml` for reproducible installs.

---

## Setup

### 1) Add your environment variables

Create a `.env` file in the project root (it exists in this repo already) with:

```bash
GOOGLE_API_KEY=your_key_here
```

This is required only if you want **Gemini-based prompt categorization**. If you already have `config/prompt_cache.json`, the pipeline can reuse it.

### 2) Put your data in the expected folders

- Video:
  - `data/testdata/test_video.mp4` (default used by the runner)
- Product catalog images:
  - `data/productimages/` containing `.jpg`, `.jpeg`, `.png`, or `.webp`

### 3) Download model weights (expected layout)

Your runner scripts currently assume these paths:

- GroundingDINO config:
  - `weights/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py`
- GroundingDINO weights:
  - `weights/GroundingDINO/weights/groundingdino_swint_ogc.pth`
- SAM weights:
  - `weights/sam-weight/sam_vit_h_4b8939.pth`

Make sure those files exist locally before running.

---

## Running the project

This repo is currently “script-first”: you run the Python files under `tests/` as entrypoints.

### End-to-end: generate prompts (if needed) + run video pipeline

Runs: Florence + Gemini → prompt cache → GroundingDINO + SAM + CLIP → outputs.

```bash
python tests/test_video_pipeline.py
```

Key defaults in `tests/test_video_pipeline.py`:

- Video: `data/testdata/test_video.mp4`
- Catalog: `data/productimages/`
- Prompt cache: `config/prompt_cache.json`
- Device: `cpu`

### Detection-only (debug GroundingDINO output)

```bash
python tests/test_detection.py
```

Outputs annotated frames in:

- `outputs/detection/`
- `outputs/crops/`

### Prompt generation only (Florence + Gemini)

```bash
python tests/test_caption_generator.py
```

This prints the prompt data and (optionally) saves it to `config/dino_prompt.json`.

The caption runner selects CUDA when available, uses FP16 on CUDA and processes
one image per batch to limit memory use. On CPU it defaults to at most four
threads (half the logical CPUs). It prints progress and timings for each batch,
releases Florence before calling Ollama, and reuses cached prompts when image
contents and generator code are unchanged. The first run still needs to load
Florence and run inference. The current clothing label cleanup uses Ollama's
`qwen2.5:3b-instruct` model.

```bash
python tests/test_caption_generator.py --refresh             # bypass cached results
python tests/test_caption_generator.py --device cpu --cpu-threads 2
python tests/test_caption_generator.py --batch-size 2         # requires more memory
```

The runner's cache is stored in `outputs/prompt_generation/cache.json`; the saved
prompt JSON retains the `clothing_prompt` and `object_prompt` fields.

---

## Outputs explained

### `outputs/video/video_timestamps.json`

The main artifact for “shoppable UX”.

- **objects**: keys are matched product names (derived from product image filename without `.jpg`)
- **clothing**: keys are GroundingDINO clothing labels
- Each list contains merged ranges computed from per-frame timestamps.

### `outputs/video/clothing_frames/`

Frames where clothing detections occurred, annotated with bounding boxes + labels.

### `outputs/video/object_frames/`

Frames where an object detection matched a catalog product, annotated with product name + score.

---

## Tuning & configuration knobs

The most important knobs live in `VideoPipeline` (`pipelines/video_pipeline.py`):

- **`target_fps`**: controls how many frames per second are processed (higher = more accurate, slower)
- **`similarity_threshold`**: CLIP similarity cutoff for a “match”
- **Detection thresholds**: in `GroundingDINODetector` (`services/detection/detector.py`)
  - `box_threshold`
  - `text_threshold`

Prompt quality matters a lot. A stronger prompt yields better detections and fewer false positives.

---

## Known limitations (honest)

- **GroundingDINO wrapper writes a `temp.jpg`** per detection call (`services/detection/detector.py`). This is simple but slower and can collide if you parallelize.
- The scripts under `tests/` are **runners**, not automated tests with assertions.
- Default runs use **CPU**; performance on longer videos will be slow unless you use CUDA and/or optimize batching.
- LLM prompt categorization expects **valid JSON** back from Gemini; it tries to strip markdown fences but can still fail on unexpected formats.

---

## Roadmap ideas (if you want to take it further)

- Add a real CLI (`python -m ... --video ... --products ... --device cuda`)
- Add `pytest` unit tests (timestamps merge, matching, segmentation correctness)
- Replace `temp.jpg` with in-memory inference / proper temp files
- Add structured logging (instead of `print`)
- Add a reproducible dependency file (`requirements.txt` or `pyproject.toml`)
- Optional: export results as chapters / WebVTT / HLS metadata for direct video player integration

---

## Credits / references

This project builds on widely used open-source models and libraries:

- GroundingDINO (text-guided object detection)
- Segment Anything (SAM)
- CLIP / OpenCLIP
- Florence-2 (captioning/OD tags)
- Google Gemini (tag normalization + categorization)

---

## License

No license file is included yet. If you’re publishing to GitHub, consider adding a license (MIT / Apache-2.0) and checking upstream model licenses for your intended use.

