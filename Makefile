.PHONY: all build test clean lint docker-build docker-up

BINARY_NAME=auditor
DOCKER_IMAGE=consistency-auditor

all: build test

build:
	go build -o bin/$(BINARY_NAME) ./cmd/auditor

test:
	go test -v ./...

clean:
	rm -rf bin/

lint:
	go vet ./...

docker-build:
	docker build -t $(DOCKER_IMAGE) .

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down
