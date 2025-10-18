# Quick Deploy to Docker Hub

**3-step guide to build and push to Docker Hub as `arpitsh018/axolotl-usf`**

## ⚡ Super Quick (1 Command)

```bash
# Run automated deployment script
bash integration/docker/deploy-to-dockerhub.sh latest
```

When prompted, enter your Docker Hub token: `dckr_pat_jl1uYaV3maSpzEBEkIAbQmW9xic`

**Done!** Image will be available at: `docker pull arpitsh018/axolotl-usf:latest`

---

## 📋 Manual Method (3 Steps)

### Step 1: Login to Docker Hub

```bash
echo "dckr_pat_jl1uYaV3maSpzEBEkIAbQmW9xic" | docker login -u arpitsh018 --password-stdin
```

### Step 2: Build and Push

```bash
bash integration/docker/build.sh \
  --registry docker.io/arpitsh018 \
  --tag latest \
  --push
```

### Step 3: Verify

```bash
docker pull arpitsh018/axolotl-usf:latest
```

---

## 🧪 Test Before Deploy

```bash
# 1. Build locally
bash integration/docker/build.sh

# 2. Test the image
docker run --gpus all \
  -v $(pwd)/integration/docker/examples/single-gpu-lora-test.yaml:/workspace/job_config.yaml \
  -v $(pwd)/output:/workspace/output \
  axolotl-usf:latest

# 3. If test passes, deploy
bash integration/docker/deploy-to-dockerhub.sh latest
```

---

## 🔄 Deploy Different Versions

```bash
# Deploy as v1.0.0
bash integration/docker/deploy-to-dockerhub.sh v1.0.0

# This creates TWO tags:
# - arpitsh018/axolotl-usf:v1.0.0
# - arpitsh018/axolotl-usf:latest
```

---

## 📥 Using the Published Image

After deployment, anyone can use your image:

### Pull and Run

```bash
# Pull from Docker Hub
docker pull arpitsh018/axolotl-usf:latest

# Run training
docker run --gpus all \
  -v $(pwd)/config.yaml:/workspace/job_config.yaml \
  -v $(pwd)/output:/workspace/output \
  arpitsh018/axolotl-usf:latest
```

### Quick Test

```bash
docker run --gpus all \
  -v $(pwd)/integration/docker/examples/single-gpu-lora-test.yaml:/workspace/job_config.yaml \
  -v $(pwd)/output:/workspace/output \
  arpitsh018/axolotl-usf:latest
```

---

## 🌐 Docker Hub Link

After deployment, view at: https://hub.docker.com/r/arpitsh018/axolotl-usf

---

## 🔒 Security Note

**IMPORTANT**: The token shown here should be kept secure!
- Don't commit it to Git
- Store in password manager
- Rotate regularly

For production use:
```bash
# Use environment variable
export DOCKER_TOKEN="dckr_pat_jl1uYaV3maSpzEBEkIAbQmW9xic"
bash integration/docker/deploy-to-dockerhub.sh latest
```

---

## 📊 What Gets Built

| Component | Size | Description |
|-----------|------|-------------|
| Base (axolotlai/axolotl:main) | ~10GB | Official Axolotl image |
| Integration Layer | ~500MB | Your training wrapper + tools |
| **Total** | **~10.5GB** | Final published image |

**Build Time**: 5-10 minutes (first time), 2-3 minutes (cached)

---

## ✅ Verification Checklist

After deployment:
- [ ] Image appears on Docker Hub
- [ ] Pull command works: `docker pull arpitsh018/axolotl-usf:latest`
- [ ] Image size is ~10.5GB
- [ ] Health check passes: `docker run --rm arpitsh018/axolotl-usf:latest /healthcheck.sh`
- [ ] Test training works

---

For detailed instructions, see: [BUILD_AND_DEPLOY.md](BUILD_AND_DEPLOY.md)
