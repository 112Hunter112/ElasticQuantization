.PHONY: all build test clean lint fmt docker-build docker-up help

BINARY_NAME=auditor
DOCKER_IMAGE=consistency-auditor

# Default target
all: fmt lint build test

## Build the application
build:
	@echo "Building..."
	go build -o bin/$(BINARY_NAME) ./cmd/auditor

## Run tests
test:
	@echo "Running tests..."
	go test -v ./...

## Run benchmarks (as tests)
bench:
	@echo "Running benchmarks..."
	go test -v ./pkg/benchmarks/...

## Clean build artifacts
clean:
	@echo "Cleaning..."
	rm -rf bin/

## Format code
fmt:
	@echo "Formatting..."
	go fmt ./...

## Lint code
lint:
	@echo "Linting..."
	go vet ./...

## Build Docker image
docker-build:
	docker build -t $(DOCKER_IMAGE) .

## Start services with Docker Compose
docker-up:
	docker-compose up -d

## Stop services
docker-down:
	docker-compose down

## Display help
help:
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)