"""Appearance intervals are independent of box motion; keep fractional seconds."""
import hashlib
from collections import defaultdict

from schemas.unscene import CueFile, Requirements
from services.unscene_errors import ProcessingFailure


def bounded_score(value):
    import math
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Non-finite model score")
    return max(0.0, min(1.0, value))


def sample_groups(samples, max_gap):
    """Coalesce duplicates at the same instant, preserving the strongest sample."""
    by_time = {}
    for sample in samples:
        timestamp = sample["time"]
        if timestamp not in by_time or sample["confidence"] > by_time[timestamp]["confidence"]:
            by_time[timestamp] = sample
    groups, current = [], []
    for sample in sorted(by_time.values(), key=lambda row: row["time"]):
        if current and (sample["time"] - current[-1]["time"] > max_gap + 1e-8
                        or sample["time"] > current[-1].get("endLimit", float("inf")) + 1e-8):
            groups.append(current)
            current = []
        current.append(sample)
    if current:
        groups.append(current)
    return groups


def sampled_boxes(samples, limit=200):
    if len(samples) <= limit:
        return samples
    # Preserve both ends and spread samples across the full appearance.
    return [samples[round(index * (len(samples) - 1) / (limit - 1))] for index in range(limit)]


class CueBuilder:
    def requirements(self, samples_by_type, duration, sample_period, model_version):
        types = []
        for code, samples in samples_by_type.items():
            segments = []
            for group in sample_groups(samples, sample_period * 1.5):
                start, end = group[0]["time"], min(duration, group[-1]["time"] + sample_period)
                if end > start:
                    segments.append(dict(startSeconds=start, endSeconds=end,
                                         confidence=min(row["confidence"] for row in group)))
            if segments:
                types.append(dict(typeCode=code, confidence=max(s["confidence"] for s in segments),
                                  segments=segments))
        return Requirements(modelVersion=model_version, types=types)

    def cues(self, observations, job, manifest, sample_period, model_version):
        by_product = defaultdict(list)
        for sample in observations:
            by_product[(sample["productId"], sample["typeCode"], sample["matchType"])].append(sample)
        cues = []
        for (product, code, match_type), samples in sorted(by_product.items()):
            for group in sample_groups(samples, sample_period * 1.5):
                start = max(0, group[0]["time"])
                end = min(job.episode.durationSeconds, max(
                    min(row["time"] + sample_period, row.get("endLimit", job.episode.durationSeconds))
                    for row in group))
                if end <= start:
                    continue
                identity = f"{job.jobId}:{product}:{code}:{match_type}:{start:.9f}:{end:.9f}"
                boxes = [dict(atSeconds=row["time"], **row["box"])
                         for row in sampled_boxes(group) if start <= row["time"] < end]
                cues.append(dict(cueId=hashlib.sha256(identity.encode()).hexdigest()[:32],
                                 productId=product, typeCode=code, matchType=match_type,
                                 confidence=min(row["confidence"] for row in group),
                                 startSeconds=start, endSeconds=end, boxes=boxes))
        if len(cues) > 5000:
            raise ProcessingFailure("cue_limit_exceeded", "More than 5000 cues; increase sampling/merge policy explicitly")
        return CueFile(jobId=job.jobId, episodeId=job.episode.episodeId,
                       modelVersion=model_version, cues=cues).validate_context(job, manifest)
