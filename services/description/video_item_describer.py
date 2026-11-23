"""Add a description to every detection in a VideoPipeline output.

Takes the dict returned by VideoPipeline.process_video:
    {"objects": {key: [{"time", "bbox"}, ...]}, "clothing": {label: [...]}}
and adds "description" to each entry in place, using that detection's box crop
from its own frame. Used by the main JobService and by the test scripts.
"""
import re
import time
from pathlib import Path

import cv2


def read_frames(video_path, times):
    """One sequential pass; returns ({time: frame}, fps) for the requested timestamps."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    wanted = {round(t * fps): t for t in times}
    frames, index, last = {}, 0, max(wanted) if wanted else -1
    try:
        while index <= last and capture.grab():
            if index in wanted:
                ok, frame = capture.retrieve()
                if ok:
                    frames[wanted[index]] = frame
            index += 1
    finally:
        capture.release()
    return frames, fps


def describe_detections(video_path, output, describer, object_labels=None, crop_dir=None):
    """Describe every object and clothing detection; returns {'described': n, 'failed': m}."""
    object_labels = object_labels or {}
    work = []
    for section, kind in (("objects", "object"), ("clothing", "clothing")):
        for key, entries in output.get(section, {}).items():
            for entry in entries:
                work.append(dict(section=section, kind=kind, key=key, entry=entry))
    work.sort(key=lambda item: (item["entry"]["time"], item["section"] != "clothing"))

    object_count = sum(item["kind"] == "object" for item in work)
    print("\n" + "=" * 50)
    print("ITEM DESCRIPTION")
    print("=" * 50)
    print("Model:", describer.model_name)
    print("Video:", video_path)
    print(f"Detections to describe: {len(work)} (objects {object_count}, clothing {len(work) - object_count}) "
          f"in {len({item['entry']['time'] for item in work})} frames")

    counts = dict(described=0, failed=0)
    if not work:
        print("\nItem description complete.")
        return counts
    frames, fps = read_frames(video_path, {item["entry"]["time"] for item in work})
    if crop_dir is not None:
        crop_dir = Path(crop_dir)
        crop_dir.mkdir(parents=True, exist_ok=True)
    started_all = time.perf_counter()
    current_time = None

    for position, item in enumerate(work, start=1):
        key, entry = item["key"], item["entry"]
        frame_index = round(entry["time"] * fps)
        if entry["time"] != current_time:
            current_time = entry["time"]
            print(f"\nProcessing frame {frame_index} (t={entry['time']}s)", flush=True)

        label = object_labels.get(key, key) if item["kind"] == "object" else key
        name = f"{label} ({key})" if label != key else key
        x1, y1, x2, y2 = entry["bbox"]
        frame = frames.get(entry["time"])
        crop = frame[y1:y2, x1:x2] if frame is not None else None
        size = f"{crop.shape[1]}x{crop.shape[0]}" if crop is not None and crop.size else "none"
        if crop_dir is not None and crop is not None and crop.size:
            safe_key = re.sub(r"[^A-Za-z0-9_-]+", "_", key)
            cv2.imwrite(str(crop_dir / f"frame_{frame_index}_{item['section']}_{safe_key}_{position}.png"), crop)

        print(f"E - Before description: {item['kind'].upper()} {name}, crop {size} "
              f"[{position}/{len(work)}]", flush=True)
        started = time.perf_counter()
        description = describer.describe(crop, label, item["kind"])
        elapsed = time.perf_counter() - started
        if "error" in description:
            counts["failed"] += 1
            print(f"F - After description: FAILED ({elapsed:.1f}s) - {description['error']}", flush=True)
        else:
            counts["described"] += 1
            colors = " + ".join([description.get("primary_color", "")] + description.get("secondary_colors", []))
            print(f"F - After description: added ({elapsed:.1f}s)", flush=True)
            print(f"    Color: {colors} | Pattern: {description.get('pattern', '')} | "
                  f"Material: {description.get('material', '')}", flush=True)
            print(f"    Summary: {description.get('summary', '')}", flush=True)
        entry["description"] = description

    print(f"\nItem descriptions: {counts} in {time.perf_counter() - started_all:.1f}s")
    print("\nItem description complete.")
    return counts
