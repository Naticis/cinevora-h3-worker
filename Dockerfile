# MiniMax H3 Naticis RunPod serverless worker.
# Base: the SGLang image the MiniMax-H3 cookbook uses for H100/H200/B200.
# "dev" moves; after your first good build, pin it by digest, e.g.
#   FROM lmsysorg/sglang@sha256:<digest>
FROM lmsysorg/sglang:dev

RUN pip install --no-cache-dir "runpod>=1.7,<2" "requests>=2.31" "huggingface_hub[cli]>=0.34"

WORKDIR /app
COPY handler.py /app/handler.py

# Weights live on the network volume, not in the image.
ENV H3_MODEL_PATH=/runpod-volume/MiniMax-H3 \
    H3_VARIANT=fl2va \
    PYTHONUNBUFFERED=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CMD ["python3", "-u", "/app/handler.py"]
