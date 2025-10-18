# Build and Deploy to Docker Hub

Complete guide for building the Axolotl USF Docker image and deploying it to Docker Hub.

## 📋 Overview

**Docker Hub Repository**: `arpitsh018/axolotl-usf`
**Tags**: `latest`, `v1.0.0`, etc.

## 🔐 Security Note

**IMPORTANT**: Never commit Docker Hub tokens to Git repositories!
- Store token securely (password manager, environment variable)
- Use `.env` files (git-ignored) for automation
- Rotate tokens regularly

## 🚀 Quick Deploy (3 Steps)

```bash
# 1. Login to Docker Hub
echo "dckr_pat_jl1uYaV3maSpzEBEkIAbQmW9xic" | docker login -u arpitsh018 --password-stdin

# 2. Build and push
bash integration/docker/build.sh \
  --registry docker.io/arpitsh018 \
  --tag latest \
  --push

# 3. Test the image
docker pull arpitsh018/axolotl-usf:latest
```

## 📝 Detailed Instructions

### Step 1: Build the Image

#### Option A: Using Build Script (Recommended)

```bash
# Build with default settings
bash integration/docker/build.sh

# This creates: axolotl-usf:latest
```

#### Option B: Manual Docker Build

```bash
# From project root
docker build \
  -f integration/docker/Dockerfile \
  -t axolotl-usf:latest \
  .
```

### Step 2: Login to Docker Hub

#### Option A: Interactive Login (More Secure)

```bash
# Login with prompt
docker login -u arpitsh018
# Enter token when prompted: dckr_pat_jl1uYaV3maSpzEBEkIAbQmW9xic
```

#### Option B: Command Line (For Scripts)

```bash
# Using echo (for automation)
echo "dckr_pat_jl1uYaV3maSpzEBEkIAbQmW9xic" | docker login -u arpitsh018 --password-stdin
```

#### Option C: Environment Variable (Best for CI/CD)

```bash
# Set token as environment variable
export DOCKER_TOKEN="dckr_pat_jl1uYaV3maSpzEBEkIAbQmW9xic"

# Login using variable
echo "$DOCKER_TOKEN" | docker login -u arpitsh018 --password-stdin
```

### Step 3: Tag the Image

```bash
# Tag for Docker Hub
docker tag axolotl-usf:latest arpitsh018/axolotl-usf:latest

# Optional: Add version tags
docker tag axolotl-usf:latest arpitsh018/axolotl-usf:v1.0.0
docker tag axolotl-usf:latest arpitsh018/axolotl-usf:stable
```

### Step 4: Push to Docker Hub

```bash
# Push latest tag
docker push arpitsh018/axolotl-usf:latest

# Push version tags (if created)
docker push arpitsh018/axolotl-usf:v1.0.0
docker push arpitsh018/axolotl-usf:stable
```

## 🎯 Complete Build & Push Workflow

### One-Command Deploy

Use the build script with registry and push options:

```bash
# Build and push in one command
bash integration/docker/build.sh \
  --registry docker.io/arpitsh018 \
  --tag latest \
  --push
```

This will:
1. Build the image
2. Tag as `arpitsh018/axolotl-usf:latest`
3. Push to Docker Hub

### Multi-Version Deploy

```bash
# Build and tag multiple versions
bash integration/docker/build.sh --tag v1.0.0

# Tag for Docker Hub with multiple versions
docker tag axolotl-usf:v1.0.0 arpitsh018/axolotl-usf:v1.0.0
docker tag axolotl-usf:v1.0.0 arpitsh018/axolotl-usf:latest
docker tag axolotl-usf:v1.0.0 arpitsh018/axolotl-usf:stable

# Push all tags
docker push arpitsh018/axolotl-usf:v1.0.0
docker push arpitsh018/axolotl-usf:latest
docker push arpitsh018/axolotl-usf:stable
```

### Build Different Axolotl Versions

```bash
# Build with Axolotl main (default)
bash integration/docker/build.sh --registry docker.io/arpitsh018 --tag latest --push

# Build with Axolotl nightly
bash integration/docker/build.sh \
  --axolotl-version nightly \
  --registry docker.io/arpitsh018 \
  --tag nightly \
  --push
```

## 📥 Using the Published Image

### Pull from Docker Hub

```bash
# Pull latest version
docker pull arpitsh018/axolotl-usf:latest

# Pull specific version
docker pull arpitsh018/axolotl-usf:v1.0.0
```

### Run Training

```bash
# Single GPU training
docker run --gpus all \
  -v $(pwd)/integration/docker/examples/single-gpu-lora-test.yaml:/workspace/job_config.yaml \
  -v $(pwd)/output:/workspace/output \
  arpitsh018/axolotl-usf:latest
```

### Update Docker Compose Files

Update the image reference in `docker-compose.yml`:

```yaml
services:
  axolotl-trainer:
    image: arpitsh018/axolotl-usf:latest  # Instead of axolotl-usf:latest
    # ... rest of config
```

## 🔄 Update Workflow

When you make changes to the integration layer:

```bash
# 1. Make your changes
vim integration/config/manager.py

# 2. Build new version
bash integration/docker/build.sh --tag v1.1.0

# 3. Test locally
docker run --gpus all -it arpitsh018/axolotl-usf:v1.1.0 /bin/bash

# 4. Tag and push
docker tag axolotl-usf:v1.1.0 arpitsh018/axolotl-usf:v1.1.0
docker tag axolotl-usf:v1.1.0 arpitsh018/axolotl-usf:latest
docker push arpitsh018/axolotl-usf:v1.1.0
docker push arpitsh018/axolotl-usf:latest
```

## 🤖 Automation Script

Create a deployment script `deploy.sh`:

```bash
#!/bin/bash
# deploy.sh - Automated build and deploy

set -e

VERSION=${1:-latest}
DOCKER_USERNAME="arpitsh018"
IMAGE_NAME="axolotl-usf"

echo "Building version: $VERSION"

# Build image
bash integration/docker/build.sh --tag "$VERSION"

# Tag for Docker Hub
docker tag "$IMAGE_NAME:$VERSION" "$DOCKER_USERNAME/$IMAGE_NAME:$VERSION"

if [ "$VERSION" != "latest" ]; then
    docker tag "$IMAGE_NAME:$VERSION" "$DOCKER_USERNAME/$IMAGE_NAME:latest"
fi

# Push to Docker Hub
echo "Pushing to Docker Hub..."
docker push "$DOCKER_USERNAME/$IMAGE_NAME:$VERSION"

if [ "$VERSION" != "latest" ]; then
    docker push "$DOCKER_USERNAME/$IMAGE_NAME:latest"
fi

echo "✅ Deployment complete!"
echo "Image: $DOCKER_USERNAME/$IMAGE_NAME:$VERSION"
```

Usage:
```bash
# Make executable
chmod +x deploy.sh

# Deploy latest
./deploy.sh latest

# Deploy specific version
./deploy.sh v1.2.0
```

## 🧪 Testing Before Push

Always test before pushing to Docker Hub:

```bash
# 1. Build image
bash integration/docker/build.sh --tag test

# 2. Run health check
docker run --rm axolotl-usf:test /healthcheck.sh

# 3. Test training (quick test)
docker run --gpus all \
  -v $(pwd)/integration/docker/examples/single-gpu-lora-test.yaml:/workspace/job_config.yaml \
  -v $(pwd)/output:/workspace/output \
  axolotl-usf:test

# 4. If all tests pass, tag and push
docker tag axolotl-usf:test arpitsh018/axolotl-usf:latest
docker push arpitsh018/axolotl-usf:latest
```

## 📊 Image Information

### Check Image Details

```bash
# View image size
docker images arpitsh018/axolotl-usf

# Inspect image
docker inspect arpitsh018/axolotl-usf:latest

# View image history
docker history arpitsh018/axolotl-usf:latest
```

### Expected Image Size

- **Base Axolotl**: ~10GB
- **Integration Layer**: ~500MB
- **Total**: ~10.5GB

## 🌐 Docker Hub Management

### View on Docker Hub

Visit: https://hub.docker.com/r/arpitsh018/axolotl-usf

### Update Description

1. Go to Docker Hub repository
2. Click "Edit"
3. Add description:

```markdown
# Axolotl USF - Automated Fine-Tuning

Production-ready Docker image for automated fine-tuning with Axolotl.

## Features
- Built on official axolotlai/axolotl image
- Integrated monitoring (W&B, MLflow, TensorBoard)
- Multi-cloud storage support (S3, Azure, GCS)
- Health checks and lifecycle management
- Single-node and multi-node training support

## Quick Start
```bash
docker pull arpitsh018/axolotl-usf:latest
docker run --gpus all -v ./config.yaml:/workspace/job_config.yaml arpitsh018/axolotl-usf:latest
```

Documentation: https://github.com/your-org/axolotl-usf
```

### Tags Strategy

Maintain these tags:
- `latest` - Latest stable release
- `v1.0.0`, `v1.1.0`, etc. - Specific versions
- `stable` - Last known good version
- `nightly` - Built with Axolotl nightly (optional)

## 🔒 Security Best Practices

1. **Never commit tokens**:
   ```bash
   # Add to .gitignore
   echo "*.token" >> .gitignore
   echo ".env.docker" >> .gitignore
   ```

2. **Use environment variables**:
   ```bash
   # Create .env.docker (git-ignored)
   echo "DOCKER_TOKEN=dckr_pat_jl1uYaV3maSpzEBEkIAbQmW9xic" > .env.docker

   # Use in scripts
   source .env.docker
   echo "$DOCKER_TOKEN" | docker login -u arpitsh018 --password-stdin
   ```

3. **Rotate tokens regularly**:
   - Generate new token every 90 days
   - Revoke old tokens
   - Update in secure storage

4. **Use read-only tokens when possible**:
   - For pulling images
   - In CI/CD pipelines

## 📚 Related Documentation

- [Docker README](README.md) - Main Docker documentation
- [Build Script](build.sh) - Automated build tool
- [Single GPU Guide](examples/SINGLE_GPU_QUICKSTART.md) - Testing guide
- [Multi-Node Setup](examples/MULTI_NODE_EXAMPLE.md) - Distributed training

---

**Ready to deploy!** 🚀
