"""Unscene AI API v1 wire contracts from the 18 October 2026 integration PDF."""
from datetime import datetime
from typing import Annotated, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Confidence = Annotated[float, Field(ge=0, le=1)]
Seconds = Annotated[float, Field(ge=0)]
Identifier = Annotated[str, Field(min_length=1, max_length=128)]
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


def media_url(value: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme not in ("https", "http") or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Expected an absolute HTTP(S) URL without embedded credentials")
    return value


class Contract(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    @model_validator(mode="after")
    def aware_times(self):
        for name in type(self).model_fields:
            value = getattr(self, name)
            if isinstance(value, datetime) and value.tzinfo is None:
                raise ValueError(f"{name} must include a timezone")
        return self


class Segment(Contract):
    startSeconds: Seconds
    endSeconds: Seconds
    confidence: Confidence | None = None

    @model_validator(mode="after")
    def ordered(self):
        if self.endSeconds <= self.startSeconds:
            raise ValueError("endSeconds must be after startSeconds")
        return self


class AllowedType(Contract):
    code: Identifier
    name: Annotated[str, Field(min_length=1)]


class Episode(Contract):
    episodeId: Identifier
    durationSeconds: Annotated[float, Field(gt=0)]
    width: Annotated[int, Field(gt=0)]
    height: Annotated[int, Field(gt=0)]
    fps: Annotated[float, Field(gt=0)]


class VideoLink(Contract):
    url: str
    expiresAt: datetime
    contentType: str
    sizeBytes: Annotated[int, Field(ge=0)]
    _url = field_validator("url")(media_url)


class CandidatesLink(Contract):
    manifestUrl: str
    expiresAt: datetime
    sha256: Sha256
    productCount: Annotated[int, Field(ge=0)]
    _url = field_validator("manifestUrl")(media_url)


class CueUpload(Contract):
    url: str
    method: Literal["PUT"]
    contentType: Literal["application/json"]
    maxBytes: Annotated[int, Field(gt=0)]
    expiresAt: datetime
    resultUrl: str
    _urls = field_validator("url", "resultUrl")(media_url)


class ClaimedJob(Contract):
    jobId: Identifier
    step: Literal["DETECT_TYPES", "MATCH"]
    leaseExpiresAt: datetime
    episode: Episode
    video: VideoLink
    taxonomyVersion: str | None = None
    allowedTypes: list[AllowedType] | None = None
    candidates: CandidatesLink | None = None
    cueUpload: CueUpload | None = None

    @model_validator(mode="after")
    def step_fields(self):
        if self.leaseExpiresAt.tzinfo is None or self.video.expiresAt.tzinfo is None:
            raise ValueError("Expiry times must include a timezone")
        if self.step == "DETECT_TYPES":
            if self.allowedTypes is None or not self.taxonomyVersion:
                raise ValueError("DETECT_TYPES needs allowedTypes and taxonomyVersion")
            codes = [item.code for item in self.allowedTypes]
            if len(codes) != len(set(codes)):
                raise ValueError("Duplicate allowed type codes")
        elif self.candidates is None or self.cueUpload is None:
            raise ValueError("MATCH needs candidates and cueUpload")
        return self


class CandidateImage(Contract):
    imageId: Identifier
    url: str
    role: str
    _url = field_validator("url")(media_url)


class CandidateProduct(Contract):
    productId: Identifier
    title: str
    brand: str | None = None
    attributes: dict = Field(default_factory=dict)
    images: list[CandidateImage] = Field(min_length=1)


class CandidateType(Contract):
    typeCode: Identifier
    segments: list[Segment]
    products: list[CandidateProduct]


class CandidateManifest(Contract):
    schemaVersion: Literal["1.0"]
    jobId: Identifier
    episodeId: Identifier
    generatedAt: datetime
    types: list[CandidateType]

    @model_validator(mode="after")
    def unique_membership(self):
        codes, products = set(), set()
        for group in self.types:
            if group.typeCode in codes:
                raise ValueError("Duplicate manifest typeCode")
            codes.add(group.typeCode)
            for product in group.products:
                if product.productId in products:
                    raise ValueError("Duplicate candidate productId")
                products.add(product.productId)
                images = [image.imageId for image in product.images]
                if len(images) != len(set(images)):
                    raise ValueError("Duplicate imageId within product")
        return self


class TypeRequirement(Contract):
    typeCode: Identifier
    confidence: Confidence
    attributes: dict = Field(default_factory=dict)
    segments: list[Segment] = Field(min_length=1, max_length=500)


class Requirements(Contract):
    schemaVersion: Literal["1.0"] = "1.0"
    modelVersion: Annotated[str, Field(min_length=1)]
    types: list[TypeRequirement] = Field(max_length=200)


class CueBox(Contract):
    atSeconds: Seconds
    x: Confidence
    y: Confidence
    w: Annotated[float, Field(gt=0, le=1)]
    h: Annotated[float, Field(gt=0, le=1)]

    @model_validator(mode="after")
    def fits_frame(self):
        if self.x + self.w > 1.000001 or self.y + self.h > 1.000001:
            raise ValueError("Box extends outside frame")
        return self


class Cue(Segment):
    cueId: Annotated[str, Field(min_length=1, max_length=64)]
    productId: Identifier
    typeCode: Identifier
    matchType: Literal["exact", "similar"]
    confidence: Confidence
    boxes: list[CueBox] = Field(default_factory=list, max_length=200)

    @model_validator(mode="after")
    def box_times(self):
        if any(not self.startSeconds <= box.atSeconds < self.endSeconds for box in self.boxes):
            raise ValueError("Box timestamp must be inside cue")
        return self


class CueFile(Contract):
    schemaVersion: Literal["1.0"] = "1.0"
    jobId: Identifier
    episodeId: Identifier
    modelVersion: Annotated[str, Field(min_length=1)]
    cues: list[Cue] = Field(max_length=5000)

    def validate_context(self, job: ClaimedJob, manifest: CandidateManifest):
        if (self.jobId, self.episodeId) != (job.jobId, job.episode.episodeId):
            raise ValueError("Cue file identity differs from claim")
        membership = {p.productId: group.typeCode for group in manifest.types for p in group.products}
        ids = set()
        for cue in self.cues:
            if cue.cueId in ids:
                raise ValueError("Duplicate cueId")
            ids.add(cue.cueId)
            if membership.get(cue.productId) != cue.typeCode:
                raise ValueError("Cue product/type is not in candidate manifest")
            if cue.endSeconds > job.episode.durationSeconds:
                raise ValueError("Cue exceeds video duration")
        return self
