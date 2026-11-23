from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Annotated, List
from urllib.parse import urlsplit


Identifier = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")]


def validate_url(value):
    parsed = urlsplit(value)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError("An absolute HTTP or HTTPS URL is required")
    return value


class Product(BaseModel):
    productId: Identifier
    imageUrl: str
    category: str
    purchaseLink: str
    imageKey:str
    additionalImageUrls: List[str] = Field(default_factory=list)

    _image_url = field_validator("imageUrl")(validate_url)

    @field_validator("additionalImageUrls")
    @classmethod
    def valid_views(cls, values):
        return [validate_url(value) for value in values]

class Episode(BaseModel):
    episodeId: Identifier
    title: str
    videoUrl: str
    _video_url = field_validator("videoUrl")(validate_url)


class AnalyzeRequest(BaseModel):
    jobId: Identifier
    seriesId: str
    callbackUrl: str
    episodes: List[Episode] = Field(min_length=1)
    products: List[Product] = Field(min_length=1)

    _callback_url = field_validator("callbackUrl")(validate_url)

    @model_validator(mode="after")
    def valid_catalog(self):
        if len({episode.episodeId for episode in self.episodes}) != len(self.episodes):
            raise ValueError("episodeId values must be unique")
        if len({product.productId for product in self.products}) != len(self.products):
            raise ValueError("productId values must be unique; use additionalImageUrls for extra views")
        objects = [product for product in self.products if product.category != "clothing"]
        if len(objects) == 1:
            raise ValueError("Object matching requires at least two distinct products for the margin check")
        return self
