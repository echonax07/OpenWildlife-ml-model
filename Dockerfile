# Stage 1: Builder – using PyTorch's devel image with build tools
FROM pytorch/pytorch:2.4.1-cuda11.8-cudnn9-devel

# To fix GPG key error when running apt-get update
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub


# Update the base OS
RUN --mount=type=cache,target="/var/cache/apt",sharing=locked \
    --mount=type=cache,target="/var/lib/apt/lists",sharing=locked \
    set -eux; \
    apt-get update; \
    apt-get upgrade -y; \
    apt install --no-install-recommends -y  \
        git; \
    apt-get autoremove -y


# Install system dependencies for opencv-python and git
RUN apt-get update  \
    && apt-get install -y libgl1 libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


ENV PYTHONUNBUFFERED=1 \
PYTHONDONTWRITEBYTECODE=1 \
PORT=${PORT:-9090} \
PIP_CACHE_DIR=/.cache \
WORKERS=1 \
THREADS=8

# Upgrade pip
RUN --mount=type=cache,target=$PIP_CACHE_DIR \
    pip install -U pip

# Install Poetry (using pip from base image)
RUN pip install "poetry==2.1.1"

ENV POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1

# Copy dependency specifications first
WORKDIR /app
COPY pyproject.toml poetry.lock ./
COPY download_bert_nltk_weights.py ./

# # # Copy SDK and application code
# COPY --from=ls_sdk / /label-studio-sdk
# COPY --from=ls_ml_backend / /label-studio-ml-backend
RUN git clone https://github.com/echonax07/OpenWildlife-ls-sdk.git /label-studio-sdk
RUN git clone https://github.com/echonax07/OpenWildlife-ls-ML-backend.git /label-studio-ml-backend


COPY . .


# # Install project dependencies
RUN poetry install

# # Final installation steps
RUN pip install mmcv==2.2.0 --no-deps -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.4/index.html 

RUN python download_bert_nltk_weights.py

# RUN echo "Hello"
# Install gdown and download checkpoint from Google Drive
RUN gdown --fuzzy "https://drive.google.com/file/d/19-n-afE2hQ4-OUB6qMCtsr4JgCXlbTxC/view?usp=sharing" -O /app/checkpoints/

# COPY /checkpoints /app/checkpoints

# copy the contents of the backend_template directory to /app
COPY /projects/LabelStudio/backend_template/. /app/



# RUN poetry install --only-root 
# # Create non-root user
# RUN useradd --create-home --shell /bin/bash appuser
# WORKDIR /app

# # # Copy virtual environment and application
# COPY --from=builder --chown=appuser:appuser /opt/venv /opt/venv
# COPY --from=builder --chown=appuser:appuser /app /app
# COPY --from=builder --chown=appuser:appuser /label-studio-sdk /label-studio-sdk
# COPY --from=builder --chown=appuser:appuser /label-studio-ml-backend /label-studio-ml-backend


# # Configure environment
# # USER appuser
# # ENV PATH="/opt/venv/bin:$PATH" \
# #     PYTHONUNBUFFERED=1 \
# #     PYTHONDONTWRITEBYTECODE=1


EXPOSE 9090

WORKDIR /app
# WORKDIR /app/projects/LabelStudio/backend_template


# RUN pip uninstall -y  opencv-python

# CMD gunicorn --preload --bind :$PORT --workers $WORKERS --threads $THREADS --timeout 0 _wsgi:app
CMD label-studio-ml start .