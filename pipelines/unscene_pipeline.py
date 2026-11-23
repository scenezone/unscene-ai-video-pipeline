"""Two-stage product analysis using the production detector, SAM and CLIP matcher."""
import math

import cv2

from postprocessing.cue_builder import CueBuilder, bounded_score
from services.unscene_errors import ProcessingFailure


class UnsceneVideoPipeline:
    def __init__(self, detector, segmenter=None, predictor=None, matcher_factory=None,
                 target_fps=1, similarity_threshold=.55, min_margin=.05,
                 single_candidate_threshold=.70, min_mask_area_ratio=.20,
                 detector_version="groundingdino-types-1", matcher_version="dino-sam-clip-1",
                 exact_verifier=None):
        if not math.isfinite(target_fps) or target_fps <= 0:
            raise ValueError("target_fps must be positive and finite")
        if not 0 <= similarity_threshold <= single_candidate_threshold <= 1:
            raise ValueError("Require 0 <= similarity threshold <= singleton threshold <= 1")
        if not math.isfinite(min_margin) or not 0 <= min_margin <= 2:
            raise ValueError("min_margin must be finite and between 0 and 2")
        if not 0 <= min_mask_area_ratio <= 1:
            raise ValueError("min_mask_area_ratio must be a fraction")
        self.detector, self.segmenter, self.predictor = detector, segmenter, predictor
        self.matcher_factory = matcher_factory
        self.target_fps, self.threshold, self.min_margin = target_fps, similarity_threshold, min_margin
        self.single_threshold, self.mask_ratio = single_candidate_threshold, min_mask_area_ratio
        self.detector_version, self.matcher_version = detector_version, matcher_version
        # A verifier must establish identity independently of the CLIP ranking.
        self.exact_verifier = exact_verifier
        self.builder = CueBuilder()
        self.sample_period = 1 / target_fps

    def frames(self, video_path, job, check):
        capture = cv2.VideoCapture(str(video_path))
        try:
            if not capture.isOpened():
                raise ProcessingFailure("decode_error", "Could not open the video stream")
            fps = capture.get(cv2.CAP_PROP_FPS)
            if not math.isfinite(fps) or fps <= 0:
                raise ProcessingFailure("decode_error", "Video frame rate is invalid")
            interval = max(1, round(fps / self.target_fps))
            self.sample_period = interval / fps
            frame_index, decoded = 0, 0
            expected_frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
            while frame_index / fps < job.episode.durationSeconds:
                check()
                available, frame = capture.read()
                if not available:
                    # Detect truncation; do not silently publish a partial result.
                    if expected_frames > 0 and frame_index + 1 < expected_frames:
                        raise ProcessingFailure("decode_error", "Video stream ended before its final frame")
                    if frame_index / fps + max(self.sample_period, .25) < job.episode.durationSeconds:
                        raise ProcessingFailure("decode_error", "Video is shorter than episode metadata")
                    break
                decoded += 1
                if frame_index % interval == 0:
                    h, w = frame.shape[:2]
                    if (w, h) != (job.episode.width, job.episode.height):
                        raise ProcessingFailure("media_changed", "Video dimensions differ from claimed episode")
                    yield frame_index / fps, frame
                frame_index += 1
            if decoded == 0:
                raise ProcessingFailure("decode_error", "Video contains no decodable frames")
        finally:
            capture.release()

    @staticmethod
    def prompt(name):
        return name.replace(".", " ").strip() + " ."

    def detect_types(self, video_path, job, check):
        samples = {kind.code: [] for kind in job.allowedTypes}
        if job.allowedTypes:
            frames = self.frames(video_path, job, check)
            try:
                for timestamp, frame in frames:
                    for kind in job.allowedTypes:
                        check()
                        # One named type per invocation makes code assignment unambiguous.
                        detections = self.detector.detect(frame, self.prompt(kind.name))
                        check()
                        scores = [bounded_score(det.score) for det in detections]
                        if scores:
                            samples[kind.code].append(dict(time=timestamp, confidence=max(scores)))
            finally:
                frames.close()
        return self.builder.requirements(samples, job.episode.durationSeconds,
                                         self.sample_period, self.detector_version)

    @staticmethod
    def pixel_box(detection, width, height):
        cx, cy, bw, bh = map(float, detection.bbox)
        if not all(math.isfinite(value) for value in (cx, cy, bw, bh)) or bw <= 0 or bh <= 0:
            return None
        x1, y1 = max(0, min(width, int((cx - bw / 2) * width))), max(0, min(height, int((cy - bh / 2) * height)))
        x2, y2 = max(0, min(width, int((cx + bw / 2) * width))), max(0, min(height, int((cy + bh / 2) * height)))
        return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None

    def _match_crop(self, frame, bbox, matcher, check):
        x1, y1, x2, y2 = bbox
        boxed = frame[y1:y2, x1:x2].copy()
        masked, area_ratio = self.segmenter.segment_with_area_ratio(frame, bbox)
        check()
        use_mask = masked.size > 0 and area_ratio >= self.mask_ratio
        crops = [masked, boxed] if use_mask else [boxed]
        for crop in crops:
            check()
            matched, product, score = matcher.match(
                cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), threshold=self.threshold,
                min_margin=self.min_margin, single_candidate_threshold=self.single_threshold)
            check()
            if matched:
                return product, bounded_score(score), crop
        return None

    def match(self, video_path, job, manifest, catalogs, check):
        matchers = {}
        for group in manifest.types:
            check()
            if group.typeCode in catalogs:
                matcher = self.matcher_factory()
                matcher.load_product_catalog(catalogs[group.typeCode], check=check)
                check()
                matchers[group.typeCode] = matcher
        observations = []
        if matchers:
            frames = self.frames(video_path, job, check)
            try:
                for timestamp, frame in frames:
                    sam_ready = False
                    for group in manifest.types:
                        check()
                        active = [s for s in group.segments if s.startSeconds <= timestamp < s.endSeconds]
                        if not active or group.typeCode not in matchers:
                            continue
                        prompt = self.prompt(group.typeCode.rsplit(".", 1)[-1].replace("_", " "))
                        detections = self.detector.detect(frame, prompt)
                        check()
                        if detections and not sam_ready:
                            self.predictor.set_image(frame, image_format="BGR")
                            check()
                            sam_ready = True
                        for detection in detections:
                            h, w = frame.shape[:2]
                            bbox = self.pixel_box(detection, w, h)
                            if bbox is None:
                                continue
                            result = self._match_crop(frame, bbox, matchers[group.typeCode], check)
                            if result is None:
                                continue
                            product, similarity, crop = result
                            # Enforce membership at observation time and again at serialization.
                            candidates = {p.productId: p for p in group.products}
                            if product not in candidates:
                                raise ProcessingFailure("invalid_match", "Matcher returned a product outside its candidate type")
                            exact = bool(self.exact_verifier and self.exact_verifier(crop, candidates[product]))
                            check()
                            x1, y1, x2, y2 = bbox
                            observations.append(dict(productId=product, typeCode=group.typeCode,
                                matchType="exact" if exact else "similar", time=timestamp,
                                endLimit=min(job.episode.durationSeconds, max(s.endSeconds for s in active)),
                                confidence=min(bounded_score(detection.score), similarity),
                                box=dict(x=x1 / w, y=y1 / h, w=(x2 - x1) / w, h=(y2 - y1) / h)))
            finally:
                frames.close()
        return self.builder.cues(observations, job, manifest, self.sample_period, self.matcher_version)
