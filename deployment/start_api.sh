#!/bin/bash

cd /workspace/unscene-ai-video-pipeline

source venv/bin/activate

export OLLAMA_HOST=http://127.0.0.1:11434

uvicorn main:app \
    --host 0.0.0.0 \
    --port 8000
