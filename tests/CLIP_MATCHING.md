# CLIP matching audit

## Current main pipeline: mask-first with box fallback

The main `VideoPipeline` and `PipelineFactory` now default to `crop_mode="hybrid"`.
For each detection independently, SAM receives BGR frames and CLIP matches the
masked crop first. A rejected mask triggers matching of that same box without
the mask. Both attempts use similarity 0.55 and product margin 0.05 by default.
When the SAM mask covers less than the default 20% of the detection box, the
pipeline skips the masked-crop CLIP match and scores the box directly. At or
above that coverage, the mask-first fallback behavior applies. Set
`min_mask_area_ratio=0` on `VideoPipeline` or `PipelineFactory` to disable this
selection rule. An accepted mask never triggers a box match. `pipeline_gelu`
remains the main activation. Explicit `box` and `masked` modes remain available.

The main `object_matching_report.json` preserves `mask_attempt`, `box_attempt`,
their rankings, scores, margins, rejection reasons, and separate crop paths.
`outcome` identifies mask acceptance, fallback acceptance, or rejection by both.
Raw top-pick disagreement counts explicitly cover fallback cases only; unlike
the test experiment, production does not compute a box-only baseline after a
mask acceptance. No claim of agreement across all detections is made.
All accepted detections, including isolated matches, retain timestamps. Rejected
attempts stay in the diagnostic report and do not become product timestamps.
Run `tests/test_video_pipeline.py` to exercise this main-pipeline configuration.

The sections below describe earlier integration stages and the separate
experiment, whose box-only baseline remains useful for full comparison.

## Test-only mask-first fallback experiment

The experiment also checks SAM mask area inside the detection box before CLIP
matching. Below **20%** coverage it scores the box directly and skips masked-crop
matching. At or above 20%, mask-first matching and the existing rejection-driven
box fallback remain unchanged. Every selected crop still requires similarity
>= 0.55 and product margin >= 0.05.

Use `--min-mask-area-ratio 0.20` to configure this cutoff, or `0` to disable it.
20% is an initial heuristic for sparse masks, not a threshold calibrated on
labeled examples. Coverage counts actual SAM mask pixels inside the clipped
box, including black foreground pixels; it adds no model inference.

Reports record the area ratio, cutoff, selected crop, and selection reason.
Direct-box outcomes are `matched_box_small_mask` or `rejected_box_small_mask`;
the box attempt retains the original similarity/margin rejection reason.
`mask_attempt` is null when skipped, and `mask_scored` gives the denominator
for mask acceptance counts. Disagreements cover only rows with scored masks.
New checkpoints preserve this policy on resume; older checkpoints without
the area setting retain their original mask-first behavior. Start a new run
to enable the area check for older data rather than estimating from RGB crops.

If scoring was interrupted after segmentation, resume from its checkpoint:

```powershell
..\venv\Scripts\python.exe tests/test_clip_mask_fallback.py --resume outputs/clip_mask_fallback/RUN_ID/report.json
```

Resume uses the saved settings and copied catalog images, skips DINO/SAM and
already-scored detections, and loads only CLIP. Checkpoint writes retry temporary
Windows file locks. A persistent lock leaves the previous JSON intact and
preserves the latest JSON in a uniquely named `.part` file; that file can also
be passed to `--resume`. Close programs holding the file before retrying.

Run from the project root:

```powershell
..\venv\Scripts\python.exe tests/test_clip_mask_fallback.py
```

This leaves the main pipeline unchanged. Defaults: complete video at 1 FPS,
`accessories`, detection thresholds 0.35/0.30, `pipeline_gelu`, corrected SAM
BGR handling, and similarity >= 0.55 plus product margin >= 0.05 for both
attempts. Multiple catalog views are pooled by product ID. Use
`--activation quickgelu` to compare that activation separately, or `--samples 8`
for a small trial. DINO, SAM, and CLIP are loaded in separate stages to limit
GPU memory usage.

After the area check selects a sufficiently large mask, the decision uses the
mask first. Only a rejected mask enables box fallback.
An accepted mask is never overridden by the box candidate. Box scores are also
computed independently for a box-only baseline, including when fallback was
not needed; these are clearly identified as baseline-only scores in the review.

`outputs/clip_mask_fallback/<run>/report.json` and `review.html` show:
- Matched via mask, matched via box fallback, and rejected by both.
- Both attempts' scores, product rankings, margins, and rejection reasons.
- Mask-only, box-only, and combined acceptance counts, plus accepted product
  disagreements between mask and box.
- Source frames, both crops, and catalog reference images for manual review.

The experiment records raw detections without temporal filtering. It does not
prove improved accuracy; manually verify fallback matches and disagreements.
Model errors fail the run rather than silently becoming a fallback match.

Checks without model inference:

```powershell
..\venv\Scripts\python.exe -m unittest discover -s tests -p test_clip_mask_fallback_checks.py -v
```

## Main pipeline integration (2026-10-06)

Update: the main CLIP loader explicitly selects `pipeline_gelu`
(`force_quick_gelu=False`). Object timestamps now retain every accepted box,
including isolated single-frame matches and multiple boxes for a product in
one frame. Clothing still uses the existing short-track filter. This changes
object timestamp counts compared with earlier filtered reports.

`outputs/video/object_matching_report.json` records every object detection,
including low-margin and low-similarity rejections, full product rankings,
scores, margins, timestamps, boxes, and links to saved crops. Invalid crops are
listed as comparison unavailable. Crop images live in a unique subdirectory
under `detection_crops` for each run. Rejected candidates are not inserted into
the confirmed-product timestamp JSON. Reports cover sampled frames (1 FPS by
default), not every source-video frame. Rerun the pipeline to generate them;
existing outputs are not retroactively changed.

The main pipeline factory now uses `ProductCatalogMatcher` from
`services/matching/product_catalog_matcher.py`. It reads the existing
`objects/by_product/OBJnnn_name` layout (or product folders directly under
the supplied root), pools reference views by product ID using maximum cosine,
and requires similarity >= 0.55 and margin >= 0.05. Flat image catalogs remain
supported, with each filename a separate candidate. No data images were moved.

`VideoPipeline` now defaults to `crop_mode="box"`: CLIP receives the RGB
bounding-box crop without a SAM mask. Set `crop_mode="masked"` on the pipeline
or factory to use segmentation; this mode explicitly passes BGR input to SAM.
The test audit
still explicitly selects its crop mode and original/corrected SAM adapter so
historical comparisons remain available.

Run the local main-pipeline entry point from the project root:

```powershell
..\venv\Scripts\python.exe tests/test_video_pipeline.py
```

## Similarity plus minimum product margin

The audit now defaults to accepting only when **best cosine >= 0.55 AND
best cosine - second-best product cosine >= 0.05**. Each product's score is
the maximum over its enabled views; two views of one product do not compete
for the margin. The decision is returned to the video pipeline, so rejected
matches do not become annotated matches or timestamps. SAM remains enabled
and the complete video remains the default.

```powershell
..\venv\Scripts\python.exe tests/test_clip_matching_audit.py
..\venv\Scripts\python.exe tests/test_clip_matching_audit.py --min-margin 0.10
```

Use `--min-margin 0` to disable the margin rule for comparison. `--baseline`
changes only model activation and SAM color handling; combine it with
`--min-margin 0` to reproduce the former threshold-only acceptance rule.
A catalog with only one product is rejected as `no_second_product` when the
margin rule is enabled. Boundary comparisons allow 1e-7 floating-point tolerance.

Each `matches.jsonl`/report row records `similarity_only_matched`, `matched`,
`second_product`, `second_similarity`, `margin`, `min_margin`, `margin_passed`
and `decision_reason`. Reasons include `below_similarity_threshold`,
`below_min_margin`, `no_second_product`, and `accepted`. Final reports also
include `similarity_only_accepted_crops` and `accepted_by_min_margin` for 0,
0.05, 0.10 and the selected margin, computed from the same raw crop scores.
These counts precede duplicate removal and temporal filtering; they are not
accuracy measurements.

Compare an existing audit without loading any models:

```powershell
..\venv\Scripts\python.exe tests/compare_clip_margins.py outputs/clip_matching_audit/<run>/report.json
```

The original report stays unchanged; the comparison is saved under
`outputs/clip_margin_comparison/`. It uses the source report's catalog and
similarity thresholds. Old flat-folder reports still compare filenames;
use grouped-catalog reports to compare product IDs.

`test_clip_matching_comparison.py` also supports `--min-margin` (default 0.05)
and now recognizes grouped product folders. Its JSON/HTML show rejection
reasons, margins, and threshold-only versus combined acceptance counts.

The audit defaults to the current product folders in
`data/productimages/objects/by_product`, with no JSON file required:

```powershell
..\venv\Scripts\python.exe tests/test_clip_matching_audit.py
```

The current folders contain 16 views in five user-assigned product IDs
(all five bag photos remain in OBJ002; the keychain is now OBJ006). See
`data/productimages/objects/by_product/README.md` and `index.html` for grouping
decisions and the visual catalog. Use `--validate-catalog` to check image paths
and list names without inference. Names are printed alongside IDs and included
as `product_name` in each audit result. Paths resolve from the project root,
even when the file is launched from another working directory.
Explicit `--catalog path/to/catalog.json` remains supported for enabled-view
curation; directory mode uses all images in current ID folders. To use a legacy
flat image folder, specify it with `--products path/to/flat_folder`.

The audit uses the existing `VideoPipeline`, `SAMSegmenter` and `CLIPMatcher`.
All changes are under `tests/`; production files remain untouched. By default,
the test-only CLIP loader enables QuickGELU to match the OpenAI weights and
SAM segmentation produces masked object crops before CLIP catalog matching.
A SAM adapter explicitly declares incoming OpenCV frames as BGR. `--baseline`
reproduces the original activation and SAM color behavior. The matching
decision still delegates to production and records the same embedding's ranking.

## How the pipeline works

1. GroundingDINO detects objects with the combined `object_prompt`. The audit
   defaults to box threshold **0.35** and text threshold **0.30**.
2. SAM masks each box, blacks out the background and crops to the box.
3. The crop is converted from OpenCV BGR to RGB. OpenCLIP's pretrained OpenAI
   **ViT-L-14** preprocessing and image encoder produce a unit-normalized vector.
4. Catalog images are independently encoded and normalized. The dot product
   with each catalog embedding is cosine similarity. This is image-to-image
   matching: detector labels and text embeddings do not influence the ranking.
5. The highest score wins if it is **>= 0.55** (the pipeline default).
   `CLIPMatcher.match()` itself defaults to 0.50, as does the older `test_clip.py`.
   A rejected match still returns the closest filename; that is not an acceptance.
6. For each frame the pipeline keeps only the highest-scoring crop per product.
   Finally it discards timestamp groups spanning less than one second, with
   gaps greater than 1.5 seconds splitting groups. An accepted crop may therefore
   be absent from the final timestamp JSON. Clothing bypasses CLIP entirely.

## Run from the project root (PowerShell)

```powershell
..\venv\Scripts\python.exe tests/test_clip_matching_audit.py
```

This executes the object branch of the actual pipeline for the complete video
at its default 1 sampled frame/second. Use `--max-duration 10` to limit a run
to the first 10 seconds.
Use `--device cpu` or `--device cuda` to override automatic selection.
The default is `--crop-mode masked`, following the original pipeline's
detection -> SAM segmentation -> RGB crop -> CLIP matching flow. Grouped
catalog matching still returns product IDs and readable names. Use
`--crop-mode box` only when you want to skip SAM for a comparison.
`--baseline` retains masked crops and reproduces the original activation/color
behavior. A supplied `--image` is treated as an existing crop and skips
video detection and SAM.

To inspect an existing segmented crop without rerunning detection/SAM:

```powershell
..\venv\Scripts\python.exe tests/test_clip_matching_audit.py --image outputs/segmentation/seg_0.png
```

Pass an existing path if that example crop does not exist. `--products` selects
a flat catalog folder; `--similarity-threshold` changes only acceptance, not
ranking. Relative input/output paths resolve against the project root.

Each run creates `outputs/clip_matching_audit/<run>/` with:

- `report.json`: settings, catalog, all crop decisions, full rankings, top-two
  margins, timestamps before temporal filtering and final pipeline timestamps.
- `crops/crop_*.png`: exact RGB inputs to CLIP saved as viewable PNGs. Crop paths
  in the report are relative to `crops/`.
- `crops/matches.jsonl`: incrementally saved decisions, including rejected ones.
- `pipeline/`: the original pipeline's annotated frames and timestamp JSON.

## Interpretation and current limitations

Cosine similarity is not a calibrated probability or measured accuracy. A high
score or narrow top-two margin can reflect visually similar products rather
than correct identity. Check saved crops against catalog images and use labeled
examples before choosing a different similarity threshold.

The current pipeline calls `sam_predictor.set_image(frame)` with an OpenCV BGR
frame but no explicit `image_format`; SAM defaults to RGB. This audit preserves
that behavior only with `--baseline`; otherwise it supplies `image_format="BGR"`.
CLIP itself receives RGB correctly in both modes.
Catalog images retain their backgrounds while crops have black backgrounds;
this preprocessing difference may affect matching. Catalog iteration is not
sorted, so exact ties resolve to whichever image was loaded first. The matcher
does not clear old embeddings on catalog reload; this runner uses a fresh matcher.
The installed OpenCLIP emits a QuickGELU mismatch warning for the loader's
`ViT-L-14` / `openai` combination (model config uses `quick_gelu=False`, while
the pretrained tag expects `True`). This deserves separate evaluation because
it may affect embeddings. The test loader passes `force_quick_gelu=True` by
default; `--baseline` passes `False` to reproduce the original configuration.

The video audit retains the combined object prompt, which can produce different
detections from the previous per-label threshold experiment. No clothing catalog
is loaded. Reports expose raw CLIP decisions before deduplication and filtering;
crop sequence numbers are not video frame numbers.

## Controlled comparison of all six variants

```powershell
..\venv\Scripts\python.exe tests/test_clip_matching_comparison.py
```

This processes the complete video at 1 sampled frame/second with box/text
thresholds 0.35/0.30 and the unchanged combined object prompt. Each detection
is computed once. The same boxes are used to save plain crops, masks from the
original SAM color handling, and masks from corrected SAM color handling.
Each of those three inputs is scored using both the original GELU configuration
and the corrected QuickGELU configuration, with the catalog re-encoded for each
model. DINO, SAM, and the two CLIP models are loaded one at a time to reduce
GPU memory use, with lossless frames preserving the exact pixels between
detection and segmentation. Use `--samples 8` for a smaller evenly spaced set across
the video, or `--max-duration 10` for an initial time window.

Outputs are saved in `outputs/clip_matching_comparison/<run>/`:

- `report.json`: settings, frame indices/timestamps, detection boxes/labels,
  all six rankings and decisions per detection, and changes relative to the
  original model plus original masked crop. Progress is saved incrementally.
- `review.html`: local side-by-side crop and score comparisons, catalog images,
  and links to source frames. Open this file in a browser after completion.
- `crops/`, `frames/`, `catalog/`: images needed to inspect the comparison.

Comparison scores are raw per-detection results before the pipeline's duplicate
removal and track filtering. The fixed threshold remains 0.55 so preprocessing
and activation changes can be isolated. Fewer acceptances or different winners
do not by themselves establish better accuracy. Label representative positive
and unknown/non-product examples before calibrating thresholds; do not select
a threshold solely from the acceptance counts.

## Observed comparison (2026-10-03)

Run `20261003_135640_084998` evaluated eight evenly spaced frames from 0 to 85
seconds of Episode 2, with one detection per selected frame. All six variants
used exactly the same detection boxes, catalog and 0.55 threshold.

| CLIP activation | Crop | Closest bag | Closest glasses | Accepted |
| --- | --- | ---: | ---: | ---: |
| Original GELU | Original SAM mask | 0 | 8 | 8 |
| Original GELU | Corrected SAM mask | 0 | 8 | 8 |
| Original GELU | Plain box | 4 | 4 | 8 |
| Corrected QuickGELU | Original SAM mask | 0 | 8 | 7 |
| Corrected QuickGELU | Corrected SAM mask | 0 | 8 | 7 |
| Corrected QuickGELU | Plain box | 5 | 3 | 7 |

Corrected plain-box matching rejected the person crop at 36 seconds, but still
accepted wrong glasses matches for the partial bag at 0 seconds and the graphic
at 85 seconds. These results favor plain boxes over black-background masks for
this sample, but do not demonstrate reliable exact product identification.
The five closest-bag rankings include the rejected person crop; the other four
are accepted bag candidates, not verified SKU matches. Neither changing the
activation nor SAM colors alone fixes the issue.
The threshold has deliberately not been tuned using this tiny sample.

Open `outputs/clip_matching_comparison/20261003_135640_084998/review.html` for
images and scores. This measured run used the original interleaved extraction
loop; subsequent code separates DINO and SAM into memory-saving stages using
lossless frames, with deterministic checks verifying pixel preservation.

Deterministic checks (no model download required):

```powershell
..\venv\Scripts\python.exe -m unittest discover -s tests -p test_clip_matching_audit_checks.py -v
```
## Decision categories

The audit JSON and console, comparison JSON and HTML, and saved-score margin
comparison JSON report separate counts for Matched, Rejected - low margin, and
Rejected - low similarity. Every crop also has a `decision_category` field.
Similarity is checked first: a score below 0.55 is low similarity even if the
margin also fails. Otherwise a margin below 0.05 is low margin; passing both
checks is matched. Matched is an acceptance decision, not verified accuracy.
Empty catalogs or missing runner-up products are reported separately as
comparison unavailable, rather than misleadingly called low similarity.

To add these counts to a previously completed comparison without model inference:

```powershell
..\venv\Scripts\python.exe tests/report_clip_categories.py outputs/clip_matching_comparison/RUN_ID/report.json
```

This writes `report_categories.json` and `review_categories.html` beside the
original report, preserving its scores and image links. Future comparison runs
include the category table directly in `review.html`.
