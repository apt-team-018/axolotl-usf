#!/bin/bash
# Quick deployment script to build and push to Docker Hub
# Usage: bash integration/docker/deploy-to-dockerhub.sh [version]

set -e

# Configuration
DOCKER_USERNAME="arpitsh018"
IMAGE_NAME="axolotl-usf"
VERSION="${1:-latest}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}Axolotl USF Docker Hub Deployment${NC}"
echo -e "${BLUE}=========================================${NC}"
echo ""
echo -e "${GREEN}Configuration:${NC}"
echo "  Docker Hub: ${DOCKER_USERNAME}/${IMAGE_NAME}"
echo "  Version: ${VERSION}"
echo ""

# Check if we're in the right directory
if [ ! -f "integration/docker/Dockerfile" ]; then
    echo -e "${RED}Error: Must run from axolotl-usf root directory${NC}"
    echo -e "${YELLOW}Current directory: $(pwd)${NC}"
    exit 1
fi

# Step 1: Login to Docker Hub
echo -e "${GREEN}Step 1: Logging in to Docker Hub...${NC}"
echo -e "${YELLOW}Note: You'll need to enter your Docker Hub token${NC}"
echo ""

# Check if token is in environment
if [ -n "$DOCKER_TOKEN" ]; then
    echo -e "${GREEN}Using token from DOCKER_TOKEN environment variable${NC}"
    echo "$DOCKER_TOKEN" | docker login -u "$DOCKER_USERNAME" --password-stdin
else
    # Prompt for token
    read -sp "Enter Docker Hub token (or Ctrl+C to cancel): " DOCKER_TOKEN
    echo ""
    echo "$DOCKER_TOKEN" | docker login -u "$DOCKER_USERNAME" --password-stdin
fi

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Docker Hub login failed${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Docker Hub login successful${NC}"
echo ""

# Step 2: Build the image
echo -e "${GREEN}Step 2: Building Docker image...${NC}"
bash integration/docker/build.sh --tag "$VERSION"

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Build failed${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Build successful${NC}"
echo ""

# Step 3: Tag for Docker Hub
echo -e "${GREEN}Step 3: Tagging image for Docker Hub...${NC}"
docker tag "${IMAGE_NAME}:${VERSION}" "${DOCKER_USERNAME}/${IMAGE_NAME}:${VERSION}"

# Also tag as latest if version is not latest
if [ "$VERSION" != "latest" ]; then
    docker tag "${IMAGE_NAME}:${VERSION}" "${DOCKER_USERNAME}/${IMAGE_NAME}:latest"
    echo -e "${GREEN}✓ Tagged as ${DOCKER_USERNAME}/${IMAGE_NAME}:${VERSION}${NC}"
    echo -e "${GREEN}✓ Tagged as ${DOCKER_USERNAME}/${IMAGE_NAME}:latest${NC}"
else
    echo -e "${GREEN}✓ Tagged as ${DOCKER_USERNAME}/${IMAGE_NAME}:${VERSION}${NC}"
fi
echo ""

# Step 4: Push to Docker Hub
echo -e "${GREEN}Step 4: Pushing to Docker Hub...${NC}"
echo -e "${YELLOW}This may take several minutes (image is ~10GB)${NC}"
echo ""

docker push "${DOCKER_USERNAME}/${IMAGE_NAME}:${VERSION}"

if [ "$VERSION" != "latest" ]; then
    docker push "${DOCKER_USERNAME}/${IMAGE_NAME}:latest"
fi

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Push failed${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}✓ Push successful${NC}"
echo ""

# Success summary
echo -e "${BLUE}=========================================${NC}"
echo -e "${GREEN}✓ Deployment Complete!${NC}"
echo -e "${BLUE}=========================================${NC}"
echo ""
echo -e "${GREEN}Image Details:${NC}"
echo "  Repository: https://hub.docker.com/r/${DOCKER_USERNAME}/${IMAGE_NAME}"
echo "  Tags pushed:"
echo "    - ${DOCKER_USERNAME}/${IMAGE_NAME}:${VERSION}"
if [ "$VERSION" != "latest" ]; then
    echo "    - ${DOCKER_USERNAME}/${IMAGE_NAME}:latest"
fi
echo ""
echo -e "${GREEN}To use this image:${NC}"
echo "  docker pull ${DOCKER_USERNAME}/${IMAGE_NAME}:${VERSION}"
echo ""
echo "  docker run --gpus all \\"
echo "    -v \$(pwd)/config.yaml:/workspace/job_config.yaml \\"
echo "    -v \$(pwd)/output:/workspace/output \\"
echo "    ${DOCKER_USERNAME}/${IMAGE_NAME}:${VERSION}"
echo ""
echo -e "${YELLOW}Quick test:${NC}"
echo "  docker run --gpus all \\"
echo "    -v \$(pwd)/integration/docker/examples/single-gpu-lora-test.yaml:/workspace/job_config.yaml \\"
echo "    -v \$(pwd)/output:/workspace/output \\"
echo "    ${DOCKER_USERNAME}/${IMAGE_NAME}:latest"
echo ""
