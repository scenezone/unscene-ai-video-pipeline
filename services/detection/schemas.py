from dataclasses import dataclass

@dataclass
class Detection:
    bbox: tuple
    score: float
    label: str