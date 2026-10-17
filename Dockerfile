# syntax=docker/dockerfile:1
# Keep NVIDIA's Blackwell-capable TensorFlow build and its matching CUDA libraries.
ARG BASE_IMAGE=nvcr.io/nvidia/tensorflow:25.02-tf2-py3@sha256:c83b37d26f19ab00d8a13cf974edd079c3d099918ec3110c304a989d6e2f75d5
FROM ${BASE_IMAGE}

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    KERAS_BACKEND=tensorflow \
    TF_USE_LEGACY_KERAS=0 \
    TF_FORCE_GPU_ALLOW_GROWTH=true \
    HTR_CONTAINER=1
WORKDIR /app
COPY docker/requirements.txt /tmp/htr-requirements.txt
# Do not install the Windows lockfile: that would replace NVIDIA TensorFlow.
RUN python -m pip install --no-cache-dir -r /tmp/htr-requirements.txt
COPY train.py ./
COPY scripts/training/ ./scripts/training/
COPY tests/ ./tests/
COPY docker/gpu_check.py ./docker/gpu_check.py
RUN mkdir -p /app/data /app/models /app/outputs \
    && python -c "import keras; from keras import ops; assert callable(ops.ctc_loss); assert callable(ops.ctc_decode)"
ENTRYPOINT ["python", "train.py"]
CMD []
