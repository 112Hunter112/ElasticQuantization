.PHONY: install test benchmark docker-build

install:
	pip install -r requirements.txt

# Run the simplified benchmark script
benchmark:
	python3 benchmarks/solver_comparison.py

# Build the Docker image
docker-build:
	docker build -t aether-neural-ode-jax:latest .

# Run locally with GPU support
docker-run:
	docker run --gpus all -p 5000:5000 -p 5001:5001 aether-neural-ode-jax:latest