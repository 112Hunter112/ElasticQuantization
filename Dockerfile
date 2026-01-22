FROM golang:1.21-alpine AS builder

WORKDIR /app

COPY go.mod ./
# COPY go.sum ./ # Uncomment when go.sum exists
RUN go mod download

COPY . .

RUN go build -o /bin/auditor ./cmd/auditor

FROM alpine:3.18

WORKDIR /app

COPY --from=builder /bin/auditor /app/auditor
COPY config/config.yaml /app/config/config.yaml

ENTRYPOINT ["/app/auditor"]
CMD ["-config", "/app/config/config.yaml"]
