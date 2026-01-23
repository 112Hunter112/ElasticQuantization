# Use standard Python image (Already contains Python 3.10 and Pip)
FROM python:3.10-slim

# Install system dependencies
# libgomp1 is CRITICAL for PyTorch/Sentence-Transformers on CPU
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Installing JAX and other dependencies
COPY requirements.txt .
# Upgrade pip and install requirements
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY neural_ode_service_jax.py .
COPY config.yaml .

COPY benchmarks ./benchmarks
COPY examples ./examples

# Environment variables
ENV XLA_PYTHON_CLIENT_PREALLOCATE=false
ENV XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
ENV JAX_ENABLE_X64=true
# Force CPU platform
ENV JAX_PLATFORMS=cpu

# Expose API and Metrics ports
EXPOSE 5000 5001

CMD ["python3", "neural_ode_service_jax.py"]