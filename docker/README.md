# Docker Setup for PSOD

This directory contains Docker configurations for development and production use of PSOD.

## 📦 Available Images

### Production Image
- **File**: `../Dockerfile`
- **Use**: Minimal production-ready image
- **Size**: ~200MB
- **Includes**: Python runtime + PSOD package

### Development Image
- **File**: `Dockerfile.dev`
- **Use**: Full development environment
- **Size**: ~1GB
- **Includes**: Python + dev tools + Jupyter + testing frameworks

## 🚀 Quick Start

### Using Docker Compose (Recommended)

```bash
# Start development environment with Jupyter
docker-compose up dev

# Run tests
docker-compose up test

# Build documentation
docker-compose up docs

# Run benchmarks
docker-compose up benchmark

# Stop all services
docker-compose down
```

### Using Make Commands

```bash
# Build production image
make docker-build

# Build development image
make docker-build-dev

# Run in production mode
make docker-run

# Start development environment
make docker-dev

# Run tests in Docker
make docker-test

# Start Jupyter in Docker
make docker-jupyter

# Clean Docker images
make docker-clean
```

### Manual Docker Commands

```bash
# Build production image
docker build -t psod:latest .

# Run production container
docker run -it --rm psod:latest

# Build development image
docker build -f docker/Dockerfile.dev -t psod:dev .

# Run development container with Jupyter
docker run -it --rm -p 8888:8888 -v $(pwd):/app psod:dev
```

## 🔧 Configuration

### Environment Variables

```bash
# Production
PYTHONUNBUFFERED=1
PYTHONDONTWRITEBYTECODE=1

# Development (additional)
JUPYTER_ENABLE_LAB=yes
PYTHONPATH=/app/src
```

### Ports

- **8888**: Jupyter Lab/Notebook
- **8000**: Documentation server
- **6006**: TensorBoard (optional)

### Volumes

Development container mounts:
- `./:/app` - Source code (read-write)
- `./data:/data` - Data directory
- `./models:/models` - Model storage
- `./logs:/logs` - Log files

## 📝 Usage Examples

### Interactive Python Shell

```bash
docker run -it --rm psod:latest python
>>> from psod import PSOD
>>> detector = PSOD()
```

### Run Example Script

```bash
docker run -it --rm \
    -v $(pwd)/examples:/examples \
    psod:latest \
    python /examples/basic_usage.py
```

### Development with Live Reload

```bash
# Start Jupyter with volume mount
docker-compose up dev

# Access at http://localhost:8888
```

### Run Specific Tests

```bash
docker run --rm \
    -v $(pwd):/app \
    psod:dev \
    pytest tests/test_core.py -v
```

### Build Documentation

```bash
docker-compose up docs

# Access at http://localhost:8000
```

## 🔍 Debugging

### Enter Running Container

```bash
# List running containers
docker ps

# Enter container
docker exec -it psod-dev /bin/bash
```

### View Logs

```bash
# View logs from all services
docker-compose logs

# View logs from specific service
docker-compose logs dev

# Follow logs
docker-compose logs -f dev
```

### Inspect Image

```bash
# View image layers
docker history psod:latest

# Inspect image
docker inspect psod:latest
```

## 🏗️ Building Custom Images

### Production with Different Base

```dockerfile
FROM python:3.9-alpine
# ... rest of Dockerfile
```

### Add Custom Dependencies

Edit `requirements.txt` or modify Dockerfile:

```dockerfile
RUN pip install \
    your-package \
    another-package
```

### Multi-platform Build

```bash
# Build for multiple platforms
docker buildx build --platform linux/amd64,linux/arm64 -t psod:latest .
```

## 🔐 Security

### Security Scanning

```bash
# Scan image for vulnerabilities
docker scan psod:latest

# Using Trivy
trivy image psod:latest
```

### Best Practices

1. **Run as non-root user** (production image)
2. **Minimal base image** (slim variant)
3. **Multi-stage builds** (smaller final image)
4. **No secrets in images** (use environment variables)
5. **Regular updates** (rebuild with latest base images)

## 🎯 Production Deployment

### Using Docker

```bash
# Pull image
docker pull your-registry/psod:latest

# Run with resource limits
docker run -d \
    --name psod-api \
    --memory="2g" \
    --cpus="2" \
    -p 8000:8000 \
    your-registry/psod:latest
```

### Using Docker Compose

```yaml
version: '3.8'
services:
  psod:
    image: your-registry/psod:latest
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '2'
          memory: 2G
    ports:
      - "8000:8000"
```

### Using Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: psod
spec:
  replicas: 3
  selector:
    matchLabels:
      app: psod
  template:
    metadata:
      labels:
        app: psod
    spec:
      containers:
      - name: psod
        image: your-registry/psod:latest
        resources:
          limits:
            memory: "2Gi"
            cpu: "2"
```

## 🧪 Testing

### Run Full Test Suite

```bash
docker-compose run --rm test
```

### Run Specific Test Category

```bash
docker-compose run --rm test pytest tests/ -m "not slow"
```

### With Coverage

```bash
docker-compose run --rm test \
    pytest tests/ --cov=src/psod --cov-report=html
```

## 📊 Performance

### Image Sizes

```bash
# Check image sizes
docker images psod

REPOSITORY   TAG       SIZE
psod         latest    ~200MB
psod         dev       ~1GB
```

### Reduce Image Size

1. Use slim/alpine base images
2. Multi-stage builds
3. Clean package caches
4. Minimize layers

### Optimize Build Time

```bash
# Use BuildKit
DOCKER_BUILDKIT=1 docker build -t psod:latest .

# Build with cache
docker build --cache-from psod:latest -t psod:latest .
```

## 🐛 Troubleshooting

### Container Won't Start

```bash
# Check logs
docker logs psod-dev

# Run with sh to debug
docker run -it --rm psod:latest /bin/sh
```

### Permission Issues

```bash
# Run as current user
docker run -it --rm \
    -u $(id -u):$(id -g) \
    -v $(pwd):/app \
    psod:dev
```

### Network Issues

```bash
# Check network
docker network ls
docker network inspect psod-network

# Recreate network
docker-compose down
docker-compose up
```

## 🔄 Maintenance

### Update Base Image

```bash
# Pull latest base
docker pull python:3.11-slim

# Rebuild image
docker-compose build --no-cache dev
```

### Clean Up

```bash
# Remove unused images
docker image prune -a

# Remove all stopped containers
docker container prune

# Remove all unused volumes
docker volume prune

# Complete cleanup
docker system prune -a --volumes
```

## 📚 Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [Multi-stage Builds](https://docs.docker.com/develop/develop-images/multistage-build/)
