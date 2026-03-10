FROM pytorch/pytorch:2.4.1-cuda11.8-cudnn9-devel

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=${PORT:-9090} \
    PIP_CACHE_DIR=/.cache \
    WORKERS=1 \
    THREADS=8 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1

# Install system dependencies for opencv-python and git
RUN apt-get update  \
    && apt-get install -y libgl1 libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml poetry.lock ./
COPY download_bert_nltk_weights.py ./
COPY sahi/ /app/sahi/

# Poetry only exists in builder
RUN pip install poetry==2.1.1

# Install Python deps (cached unless lockfile changes)
RUN --mount=type=cache,target=/pip-cache \
    poetry install --no-root

# Download BERT and NLTK weights
RUN python download_bert_nltk_weights.py

# Download checkpoint from Google Drive
RUN gdown --fuzzy "https://drive.google.com/file/d/1WkDy4fP1xqaDVfMywZGWlZBOvK5BiMxt/view?usp=sharing" -O /app/checkpoints/

# CUDA-specific mmcv wheel
RUN --mount=type=cache,target=/pip-cache \
    pip install mmcv==2.2.0 --no-deps \
      -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.4/index.html

# App code LAST (fast iteration)
COPY . .
COPY /projects/LabelStudio/backend_template/. /app/
COPY /projects/ /app/projects/

EXPOSE 9090

CMD ["label-studio-ml", "start", "."]
