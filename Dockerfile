# syntax=docker/dockerfile:1
# GA-SiG runtime image: PyTorch 2.8.0 + CUDA 12.6 + Cantera 3.2.0 on Python 3.11.
#
# Build:  docker build -t ga-sig .
# Run:    docker run --rm --gpus all -v "<run folder>:/workspace" ga-sig run config.json
# CPU:    docker run --rm -v "<run folder>:/workspace" ga-sig run config.json --device cpu

FROM pytorch/pytorch:2.8.0-cuda12.6-cudnn9-runtime AS deps

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Versions match the tested environment in README.md.
RUN pip install "numpy==1.26.4" "cantera==3.2.0" "matplotlib>=3.7" "pytest>=7"

FROM deps AS app

WORKDIR /opt/ga-sig
COPY pyproject.toml README.md LICENSE ./
COPY gasig ./gasig
RUN pip install --no-deps .

# The run folder (config.json, mechanism, data) is mounted here.
WORKDIR /workspace
ENTRYPOINT ["python", "-m", "gasig"]
CMD ["--help"]
