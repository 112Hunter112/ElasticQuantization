FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Install Python
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install JAX and dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY neural_ode_service_jax.py .
COPY config.yaml .
# Copy subdirectories if they exist (optional)
# COPY benchmarks ./benchmarks
# COPY examples ./examples

# Environment variables for JAX optimization
ENV XLA_PYTHON_CLIENT_PREALLOCATE=false
ENV XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
ENV JAX_ENABLE_X64=true

# Expose API and Metrics ports
EXPOSE 5000 5001

CMD ["python3", "neural_ode_service_jax.py"]