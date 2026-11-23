import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parents[1])
)
import json

from services.model_registry import ModelRegistry
from services.pipeline_factory import PipelineFactory
from services.job_service import JobService


with open(
    r"C:\Users\somya\Downloads\AI_TEST_PAYLOAD.json",
    "r"
) as f:
    payload_file = json.load(f)

payload = payload_file[
    "analyzeJobRequest"
]

registry = ModelRegistry()

registry.load_models()

factory = PipelineFactory(
    registry
)

job_service = JobService(
    factory,
    registry
)

result = job_service.process_job(
    payload
)

print(result)