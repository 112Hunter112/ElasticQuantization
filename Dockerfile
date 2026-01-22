# Use the standard Go image (Debian-based) which already includes git and build tools
FROM golang:1.21 AS builder

WORKDIR /app

# Copy dependency files
COPY go.mod ./
# COPY go.sum ./ 

# Download dependencies (git is already installed in this base image)
RUN go mod download

# Copy the rest of the source code
COPY . .

# Build the binary
RUN CGO_ENABLED=0 GOOS=linux go build -o /bin/auditor ./cmd/auditor

# Use a minimal runtime image
FROM alpine:3.18

WORKDIR /app

# Copy the binary from the builder
COPY --from=builder /bin/auditor /app/auditor
COPY config/config.yaml /app/config/config.yaml

ENTRYPOINT ["/app/auditor"]
CMD ["-config", "/app/config/config.yaml"]