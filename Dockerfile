# Stage 1: Builder – using PyTorch's devel image with build tools
FROM pytorch/pytorch:2.4.1-cuda11.8-cudnn9-devel AS builder

# Install system dependencies (venv and build essentials)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry (using pip from base image)
RUN pip install "poetry==1.8.4"

# Create and activate virtual environment
RUN python -m venv --copies /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
ENV POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1

# Copy dependency specifications first
WORKDIR /app
COPY pyproject.toml poetry.lock ./


# Copy SDK and application code
COPY --from=ls_sdk / /label-studio-sdk
COPY --from=ls_ml_backend / /label-studio-ml-backend
COPY . .



# Install project dependencies
RUN poetry install --without dev

# Final installation steps
RUN poetry install --without dev && \
    pip install mmcv==2.2.0 --no-deps -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.4/index.html 

# Stage 2: Final runtime – using PyTorch's slim runtime image
FROM pytorch/pytorch:2.4.1-cuda11.8-cudnn9-runtime AS final

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0


# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser
WORKDIR /app

# Copy virtual environment and application
COPY --from=builder --chown=appuser:appuser /opt/venv /opt/venv
COPY --from=builder --chown=appuser:appuser /app /app
COPY --from=builder --chown=appuser:appuser /label-studio-sdk /label-studio-sdk
COPY --from=builder --chown=appuser:appuser /label-studio-ml-backend /label-studio-ml-backend


# Configure environment
USER appuser
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1


EXPOSE 9090

WORKDIR /app/projects/LabelStudio/backend_template



# RUN pip uninstall -y  opencv-python

CMD gunicorn --preload --bind :$PORT --workers $WORKERS --threads $THREADS --timeout 0 _wsgi:app