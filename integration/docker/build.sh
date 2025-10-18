#!/bin/bash
# Build script for Axolotl USF Docker images
# Supports multiple build targets and versioning

set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
AXOLOTL_VERSION="${AXOLOTL_VERSION:-main}"
IMAGE_NAME="${IMAGE_NAME:-axolotl-usf}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
REGISTRY="${REGISTRY:-}"
PUSH="${PUSH:-false}"
BUILD_CONTEXT="."

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --axolotl-version)
            AXOLOTL_VERSION="$2"
            shift 2
            ;;
        --tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        --registry)
            REGISTRY="$2"
            shift 2
            ;;
        --push)
            PUSH="true"
            shift
            ;;
        --no-cache)
            NO_CACHE="--no-cache"
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --axolotl-version VERSION  Base Axolotl image version (default: main)"
            echo "  --tag TAG                  Image tag (default: latest)"
            echo "  --registry REGISTRY        Docker registry to push to"
            echo "  --push                     Push image to registry after build"
            echo "  --no-cache                 Build without using cache"
            echo "  --help                     Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --tag v1.0.0"
            echo "  $0 --axolotl-version nightly --tag nightly"
            echo "  $0 --registry ghcr.io/myorg --tag latest --push"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Construct full image name
if [ -n "$REGISTRY" ]; then
    FULL_IMAGE_NAME="${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"
else
    FULL_IMAGE_NAME="${IMAGE_NAME}:${IMAGE_TAG}"
fi

echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}Axolotl USF Docker Build${NC}"
echo -e "${BLUE}=========================================${NC}"
echo ""
echo -e "${GREEN}Build Configuration:${NC}"
echo "  Base Axolotl version: ${AXOLOTL_VERSION}"
echo "  Image name: ${FULL_IMAGE_NAME}"
echo "  Build context: ${BUILD_CONTEXT}"
echo "  Push to registry: ${PUSH}"
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed${NC}"
    exit 1
fi

# Check if we're in the correct directory
if [ ! -f "integration/docker/Dockerfile" ]; then
    echo -e "${RED}Error: Must run from axolotl-usf root directory${NC}"
    echo -e "${YELLOW}Current directory: $(pwd)${NC}"
    exit 1
fi

# Build the image
echo -e "${GREEN}Building Docker image...${NC}"
docker build \
    ${NO_CACHE} \
    --build-arg AXOLOTL_VERSION=${AXOLOTL_VERSION} \
    -f integration/docker/Dockerfile \
    -t ${FULL_IMAGE_NAME} \
    ${BUILD_CONTEXT}

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Build successful!${NC}"
    echo ""
    echo -e "${GREEN}Image details:${NC}"
    docker images ${FULL_IMAGE_NAME}
else
    echo -e "${RED}✗ Build failed!${NC}"
    exit 1
fi

# Tag with additional tags
if [ "$IMAGE_TAG" != "latest" ]; then
    echo ""
    echo -e "${GREEN}Tagging as latest...${NC}"
    if [ -n "$REGISTRY" ]; then
        docker tag ${FULL_IMAGE_NAME} ${REGISTRY}/${IMAGE_NAME}:latest
    else
        docker tag ${FULL_IMAGE_NAME} ${IMAGE_NAME}:latest
    fi
fi

# Push to registry if requested
if [ "$PUSH" = "true" ]; then
    if [ -z "$REGISTRY" ]; then
        echo -e "${RED}Error: --registry must be specified when using --push${NC}"
        exit 1
    fi

    echo ""
    echo -e "${GREEN}Pushing to registry...${NC}"
    docker push ${FULL_IMAGE_NAME}

    if [ "$IMAGE_TAG" != "latest" ]; then
        docker push ${REGISTRY}/${IMAGE_NAME}:latest
    fi

    echo -e "${GREEN}✓ Push successful!${NC}"
fi

echo ""
echo -e "${BLUE}=========================================${NC}"
echo -e "${GREEN}Build Complete!${NC}"
echo -e "${BLUE}=========================================${NC}"
echo ""
echo -e "${GREEN}To run the image:${NC}"
echo "  docker run --gpus all -v ./job_config.yaml:/workspace/job_config.yaml ${FULL_IMAGE_NAME}"
echo ""
echo -e "${GREEN}To use with Docker Compose:${NC}"
echo "  docker-compose -f integration/docker/docker-compose.yml up"
echo ""
