# Stage 1: Builder – install dependencies and build the Python environment
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 AS builder

# Install Python (>=3.9) and build essentials (compiler, etc.) for installing packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-dev python3-pip build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry (Python dependency manager) at the specified version (1.8.4)
RUN pip3 install "poetry==1.8.4"

# Create a virtual environment for the project (use --copies to avoid symlinks issues in multi-stage builds:contentReference[oaicite:0]{index=0})
RUN python3 -m venv --copies /opt/venv

# Activate the virtualenv by updating PATH (so python/pip refer to /opt/venv)
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Configure Poetry to not create its own venv (use the one we made) and to run non-interactively
ENV POETRY_VIRTUALENVS_CREATE=false POETRY_NO_INTERACTION=1

# Copy only dependency definitions first (for caching)
WORKDIR /app
COPY pyproject.toml poetry.lock ./

# Copy the label-studio-sdk repository into the image
COPY --from=ls_sdk / /label-studio-sdk

# Install only runtime dependencies into the virtualenv (no dev dependencies for production)
RUN poetry install --without dev

# Copy the actual application code into the image
COPY . . 

# (Optional) Install the current project into the venv, if it's a package (so it can be imported as a module)
RUN poetry install --without dev

RUN pip install \
      mmcv==2.2.0 \
      -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.4/index.html

# Stage 2: Final runtime – minimal image with CUDA libs and our app
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 AS final

# Install Python runtime (without dev tools) in the final image
RUN apt-get update && apt-get install -y --no-install-recommends python3-minimal \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

# Set work directory
WORKDIR /app

# Copy virtual environment from builder (contains Python interpreter and all deps)
COPY --from=builder --chown=appuser:appuser /opt/venv /opt/venv

# Copy application code from builder
COPY --from=builder --chown=appuser:appuser /app /app

# Switch to non-root user
USER appuser

# Ensure the virtualenv's Python and pip are used by default
ENV PATH="/opt/venv/bin:$PATH"

# (Optional) Python runtime optimizations
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

COPY --chown=1001:0 --from=venv-builder /label-studio-sdk /label-studio-sdk
# Set default command (modify as needed for your application)
# Example: run an inference script or start a training job
CMD ["label-studio-ml", "start", "projects/LabelStudio/backend_template", "--with", "config_file=configs/eider_ducks/mm_grounding_dino_real_filtered_epoch10"]
