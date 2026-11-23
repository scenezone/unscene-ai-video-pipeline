import cv2
import os
import json
import math
from datetime import datetime
from collections import Counter
from copy import deepcopy


DEFAULT_MIN_MASK_AREA_RATIO = 0.20


class VideoPipeline:

    def __init__(
        self,
        detector,
        segmenter,
        matcher,
        sam_predictor,
        clothing_prompt=None,
        object_prompt=None,
        output_dir="outputs/video",
        similarity_threshold=0.55,
        target_fps=1,
        max_duration=None,
        crop_mode="hybrid",
        sam_image_format="BGR",
        min_mask_area_ratio=DEFAULT_MIN_MASK_AREA_RATIO,
    ):

        self.detector = detector
        self.segmenter = segmenter
        self.matcher = matcher
        self.sam_predictor = sam_predictor
        if crop_mode not in ("box", "masked", "hybrid"):
            raise ValueError("crop_mode must be box, masked, or hybrid")
        if not math.isfinite(min_mask_area_ratio) or not 0 <= min_mask_area_ratio <= 1:
            raise ValueError("min_mask_area_ratio must be finite and between 0 and 1")
        self.crop_mode = crop_mode
        self.sam_image_format = sam_image_format
        self.min_mask_area_ratio = min_mask_area_ratio

        self.clothing_prompt = clothing_prompt
        self.object_prompt = object_prompt

        self.enable_clothing = bool(clothing_prompt and clothing_prompt.strip())
        self.enable_objects = bool(object_prompt and object_prompt.strip())

        self.similarity_threshold = similarity_threshold
        self.target_fps = target_fps
        self.max_duration = max_duration

        self.output_dir = output_dir

        self.clothing_dir = os.path.join(output_dir, "clothing_frames")
        self.object_dir = os.path.join(output_dir, "object_frames")

        os.makedirs(self.clothing_dir, exist_ok=True)
        os.makedirs(self.object_dir, exist_ok=True)

        self.product_timestamps = {}
        self.clothing_timestamps = {}

    # --------------------------------------------------
    # FILTER SHORT TRACKS (< 1 sec)
    # --------------------------------------------------

    def filter_short_tracks(self, entries, min_duration=1.0):

        if not entries:
            return []

        entries = sorted(entries, key=lambda x: x["time"])

        filtered = []
        group = [entries[0]]

        for i in range(1, len(entries)):

            if entries[i]["time"] - entries[i - 1]["time"] <= 1.5:
                group.append(entries[i])
            else:
                duration = group[-1]["time"] - group[0]["time"]

                if duration >= min_duration:
                    filtered.extend(group)

                group = [entries[i]]

        # last group
        duration = group[-1]["time"] - group[0]["time"]
        if duration >= min_duration:
            filtered.extend(group)

        return filtered

    # --------------------------------------------------

    def process_video(self, video_path):
        self.product_timestamps = {}
        self.clothing_timestamps = {}

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise RuntimeError("Cannot open video")

        detection_rows = []
        crop_folder = os.path.join("detection_crops", datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        os.makedirs(os.path.join(self.output_dir, crop_folder), exist_ok=True)

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        self.video_fps = video_fps

        frame_interval = max(1, int(video_fps / self.target_fps))

        print("Video FPS:", video_fps)
        print("Processing every", frame_interval, "frames")

        frame_idx = 0

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            current_time = frame_idx / self.video_fps

            # STOP EARLY IF NEEDED
            if self.max_duration is not None and current_time > self.max_duration:
                print(f"\nStopping at {self.max_duration} seconds")
                break

            if frame_idx % frame_interval != 0:
                frame_idx += 1
                continue

            print(f"\nProcessing frame {frame_idx} (t={round(current_time,2)}s)")
            print("A - Before clothing detection")
            h, w = frame.shape[:2]

            clothing_frame = frame.copy()
            object_frame = frame.copy()

            # --------------------------------------------------
            # CLOTHING DETECTION
            # --------------------------------------------------

            if self.enable_clothing:

                clothing_detections = self.detector.detect(
                    frame, self.clothing_prompt
                )
                print("B - After clothing detection")

                for det in clothing_detections:

                    cx, cy, bw, bh = det.bbox

                    x1 = int((cx - bw / 2) * w)
                    y1 = int((cy - bh / 2) * h)
                    x2 = int((cx + bw / 2) * w)
                    y2 = int((cy + bh / 2) * h)

                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(w, x2)
                    y2 = min(h, y2)

                    label = det.label.lower()

                    cv2.rectangle(clothing_frame, (x1, y1), (x2, y2), (255, 0, 0), 3)

                    cv2.putText(
                        clothing_frame,
                        label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 0, 0),
                        2
                    )

                    self.clothing_timestamps.setdefault(label, []).append({
                        "time": current_time,
                        "bbox": [x1, y1, x2, y2]
                    })

                if len(clothing_detections) > 0:

                    frame_path = os.path.join(
                        self.clothing_dir,
                        f"frame_{frame_idx}.jpg"
                    )

                    cv2.imwrite(frame_path, clothing_frame)
                

            # --------------------------------------------------
            # OBJECT DETECTION
            # --------------------------------------------------

            if self.enable_objects:
                print("C - Before object detection")
                object_detections = self.detector.detect(
                    frame, self.object_prompt
                )
                print("D - After object detection")

                accepted_matches = []

                if len(object_detections) > 0 and self.crop_mode in ("masked", "hybrid"):
                    if self.sam_image_format is None:
                        self.sam_predictor.set_image(frame)
                    else:
                        self.sam_predictor.set_image(frame, image_format=self.sam_image_format)

                for detection_index, det in enumerate(object_detections):

                    cx, cy, bw, bh = det.bbox

                    x1 = int((cx - bw / 2) * w)
                    y1 = int((cy - bh / 2) * h)
                    x2 = int((cx + bw / 2) * w)
                    y2 = int((cy + bh / 2) * h)

                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(w, x2)
                    y2 = min(h, y2)

                    bbox = (x1, y1, x2, y2)
                    row = dict(frame=frame_idx, time=current_time,
                               detection_index=detection_index, bbox=list(bbox),
                               detector_label=det.label, accepted=False,
                               decision_category="rejected_unavailable",
                               decision_reason="invalid_crop")
                    detector_score = getattr(det, "score", None)
                    if detector_score is not None:
                        row["detector_score"] = float(detector_score)
                    detection_rows.append(row)
                    if x2 <= x1 or y2 <= y1:
                        continue
                    mask_area_ratio = None
                    if self.crop_mode == "box":
                        segmented = frame[y1:y2, x1:x2].copy()
                    else:
                        segment_with_area_ratio = getattr(
                            type(self.segmenter), "segment_with_area_ratio", None)
                        if callable(segment_with_area_ratio):
                            segmented, mask_area_ratio = self.segmenter.segment_with_area_ratio(frame, bbox)
                        else:
                            segmented = self.segmenter.segment(frame, bbox)
                    if segmented.size == 0:
                        continue

                    prefix = os.path.join(crop_folder, f"frame_{frame_idx}_det_{detection_index}")
                    small_mask = (self.crop_mode == "hybrid" and mask_area_ratio is not None
                                  and mask_area_ratio < self.min_mask_area_ratio)
                    if small_mask:
                        selected = self.match_crop(frame[y1:y2, x1:x2].copy(), prefix + "_box.png")
                        row.update(selected, mask_attempt=None, box_attempt=selected,
                                   fallback_attempted=False, mask_area_ratio=mask_area_ratio,
                                   min_mask_area_ratio=self.min_mask_area_ratio, selected_crop="box",
                                   crop_selection_reason="mask_area_below_threshold")
                        row["outcome"] = ("matched_box_small_mask" if selected["accepted"]
                                          else "rejected_box_small_mask")
                    else:
                        primary = self.match_crop(
                            segmented,
                            prefix + ("_box.png" if self.crop_mode == "box" else "_mask.png"))
                        selected = primary
                        box_attempt = None
                        if self.crop_mode == "hybrid" and not primary["accepted"]:
                            box_attempt = self.match_crop(frame[y1:y2, x1:x2].copy(), prefix + "_box.png")
                            selected = box_attempt
                        row.update(selected)
                        row.update(mask_attempt=primary if self.crop_mode != "box" else None,
                                   box_attempt=primary if self.crop_mode == "box" else box_attempt,
                                   fallback_attempted=box_attempt is not None)
                        if self.crop_mode == "hybrid":
                            row.update(mask_area_ratio=mask_area_ratio,
                                       min_mask_area_ratio=self.min_mask_area_ratio,
                                       selected_crop="box" if box_attempt else "mask",
                                       crop_selection_reason="mask_first")
                            row["outcome"] = ("matched_mask" if primary["accepted"] else
                                              "matched_box_fallback" if selected["accepted"] else "rejected_both")
                        else:
                            row["outcome"] = ("matched_" + self.crop_mode if selected["accepted"] else "rejected")
                    matched, product, score = selected["accepted"], selected["best_product"], selected["similarity"]

                    if matched:
                        accepted_matches.append((detection_index, product, score, bbox))

                for detection_index, product, score, bbox in accepted_matches:

                    x1, y1, x2, y2 = bbox

                    # Grouped catalogs return IDs; legacy flat catalogs return image filenames.
                    product_name = (os.path.splitext(product)[0]
                                    if os.path.splitext(product)[1].lower() in (".jpg", ".jpeg", ".png", ".webp")
                                    else product)
                    display_name = getattr(self.matcher, "product_metadata", {}).get(product, {}).get("name", product_name)

                    cv2.rectangle(object_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

                    cv2.putText(
                        object_frame,
                        f"{display_name} {score:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2
                    )

                    frame_path = os.path.join(
                        self.object_dir,
                        f"frame_{frame_idx}_det_{detection_index}_{product_name}.jpg"
                    )

                    cv2.imwrite(frame_path, object_frame)

                    self.product_timestamps.setdefault(product_name, []).append({
                        "time": current_time,
                        "bbox": [x1, y1, x2, y2]
                    })

            frame_idx += 1
        print("Reached end of process_video")
        cap.release()

        # --------------------------------------------------
        # FINAL JSON (FILTERED)
        # --------------------------------------------------

        output = {
            "objects": {},
            "clothing": {}
        }

        for product, entries in self.product_timestamps.items():
            output["objects"][product] = [
                {"time": round(e["time"], 2), "bbox": e["bbox"]}
                for e in entries
            ]

        for clothing, entries in self.clothing_timestamps.items():

            filtered = self.filter_short_tracks(entries)

            if not filtered:
                continue

            output["clothing"][clothing] = [
                {"time": round(e["time"], 2), "bbox": e["bbox"]}
                for e in filtered
            ]

        json_path = os.path.join(self.output_dir, "video_timestamps.json")
        print("Writing output JSON")
        with open(json_path, "w") as f:
            json.dump(output, f, indent=4)

        categories = dict.fromkeys(("matched", "rejected_low_margin", "rejected_low_similarity",
                                    "rejected_unavailable"), 0)
        categories.update(Counter(row["decision_category"] for row in detection_rows))
        comparable = [row for row in detection_rows if row.get("fallback_attempted")]
        disagreements = sum(row["mask_attempt"]["best_product"] != row["box_attempt"]["best_product"]
                            for row in comparable)
        report = dict(video=str(video_path), status="complete",
                      settings=dict(crop_mode=self.crop_mode, object_prompt=self.object_prompt,
                                    similarity_threshold=self.similarity_threshold,
                                    min_mask_area_ratio=self.min_mask_area_ratio,
                                    min_margin=getattr(self.matcher, "min_margin", None),
                                    target_fps=self.target_fps, max_duration=self.max_duration),
                      total_detections=len(detection_rows), decision_categories=categories,
                      outcomes=dict(Counter(row.get("outcome", "invalid_crop") for row in detection_rows)),
                      raw_top_pick_comparison=dict(scope="Fallback attempts only; box is not scored after mask acceptance",
                                                   comparable=len(comparable), disagreements=disagreements),
                      detections=detection_rows)
        if not isinstance(report["settings"]["min_margin"], (int, float)):
            report["settings"]["min_margin"] = None
        with open(os.path.join(self.output_dir, "object_matching_report.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, allow_nan=False)
        print("Object matching decisions:", categories)

        print("\nTimestamp JSON saved:", json_path)
        print("\nVideo processing complete.")
        return output

    def match_crop(self, crop_bgr, relative_path):
        """Snapshot one attempt before a fallback can replace matcher state."""
        matched, product, score = self.matcher.match(
            cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB), self.similarity_threshold)
        details = getattr(self.matcher, "last_decision", None)
        attempt = deepcopy(details) if isinstance(details, dict) else dict(
            decision_category="matched" if matched else "rejected_unavailable",
            decision_reason="accepted" if matched else "matcher_rejected")
        attempt.update(accepted=bool(matched), best_product=product, similarity=float(score),
                       crop=relative_path.replace(os.sep, "/"))
        if not cv2.imwrite(os.path.join(self.output_dir, relative_path), crop_bgr):
            raise OSError(f"Cannot save detection crop: {relative_path}")
        return attempt
