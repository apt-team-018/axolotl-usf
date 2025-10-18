# Multi-Node Distributed Training Example

Complete example for running distributed training across 4 nodes with 2 GPUs per node (8 total GPUs).

## 📋 Setup Overview

**Configuration:**
- **Total Nodes**: 4
- **GPUs per Node**: 2
- **Total GPUs**: 8
- **World Size**: 8

**Node Details:**
| Node | Hostname | IP Address | Role | NODE_RANK | GPUs |
|------|----------|------------|------|-----------|------|
| Node 0 | master | 192.168.1.100 | Master | 0 | 2 |
| Node 1 | worker1 | 192.168.1.101 | Worker | 1 | 2 |
| Node 2 | worker2 | 192.168.1.102 | Worker | 2 | 2 |
| Node 3 | worker3 | 192.168.1.103 | Worker | 3 | 2 |

## 🚀 Step-by-Step Deployment

### Prerequisites

On **all 4 nodes**:

```bash
# 1. Ensure Docker and NVIDIA runtime are installed
docker --version
nvidia-smi

# 2. Pull the image
docker pull axolotl-usf:latest

# 3. Ensure network connectivity between nodes
ping 192.168.1.100  # from all nodes
ping 192.168.1.101
ping 192.168.1.102
ping 192.168.1.103

# 4. Open required ports (on all nodes)
# Port 29500: PyTorch distributed communication
sudo ufw allow 29500/tcp
```

### Node 0 (Master) - 192.168.1.100

```bash
# Terminal on master node
cd /path/to/axolotl-usf

# Set environment variables
export WORLD_SIZE=8
export GPUS_PER_NODE=2
export MASTER_ADDR=192.168.1.100
export MASTER_PORT=29500
export NODE_RANK=0

# Start master
docker-compose -f integration/docker/docker-compose.multi-node.yml up master
```

Or as a single command:
```bash
WORLD_SIZE=8 GPUS_PER_NODE=2 MASTER_ADDR=192.168.1.100 MASTER_PORT=29500 \
  docker-compose -f integration/docker/docker-compose.multi-node.yml up master
```

### Node 1 (Worker 1) - 192.168.1.101

```bash
# Terminal on worker node 1
cd /path/to/axolotl-usf

# Set environment variables
export WORLD_SIZE=8
export GPUS_PER_NODE=2
export MASTER_ADDR=192.168.1.100
export MASTER_PORT=29500
export NODE_RANK=1

# Start worker
docker-compose -f integration/docker/docker-compose.multi-node.yml up worker
```

Or as a single command:
```bash
WORLD_SIZE=8 GPUS_PER_NODE=2 MASTER_ADDR=192.168.1.100 NODE_RANK=1 \
  docker-compose -f integration/docker/docker-compose.multi-node.yml up worker
```

### Node 2 (Worker 2) - 192.168.1.102

```bash
# Terminal on worker node 2
cd /path/to/axolotl-usf

WORLD_SIZE=8 GPUS_PER_NODE=2 MASTER_ADDR=192.168.1.100 NODE_RANK=2 \
  docker-compose -f integration/docker/docker-compose.multi-node.yml up worker
```

### Node 3 (Worker 3) - 192.168.1.103

```bash
# Terminal on worker node 3
cd /path/to/axolotl-usf

WORLD_SIZE=8 GPUS_PER_NODE=2 MASTER_ADDR=192.168.1.100 NODE_RANK=3 \
  docker-compose -f integration/docker/docker-compose.multi-node.yml up worker
```

## 📝 Quick Reference

### All Commands in Order

```bash
# ============================================================================
# NODE 0 (Master - 192.168.1.100)
# ============================================================================
WORLD_SIZE=8 GPUS_PER_NODE=2 MASTER_ADDR=192.168.1.100 \
  docker-compose -f integration/docker/docker-compose.multi-node.yml up master

# ============================================================================
# NODE 1 (Worker - 192.168.1.101)
# ============================================================================
WORLD_SIZE=8 GPUS_PER_NODE=2 MASTER_ADDR=192.168.1.100 NODE_RANK=1 \
  docker-compose -f integration/docker/docker-compose.multi-node.yml up worker

# ============================================================================
# NODE 2 (Worker - 192.168.1.102)
# ============================================================================
WORLD_SIZE=8 GPUS_PER_NODE=2 MASTER_ADDR=192.168.1.100 NODE_RANK=2 \
  docker-compose -f integration/docker/docker-compose.multi-node.yml up worker

# ============================================================================
# NODE 3 (Worker - 192.168.1.103)
# ============================================================================
WORLD_SIZE=8 GPUS_PER_NODE=2 MASTER_ADDR=192.168.1.100 NODE_RANK=3 \
  docker-compose -f integration/docker/docker-compose.multi-node.yml up worker
```

## 🔍 Verification

### Check Container Status

On each node:
```bash
# View running containers
docker ps

# Check container logs
docker logs -f <container_name>

# Check GPU usage
docker exec <container_name> nvidia-smi
```

### Verify Network Communication

From master node:
```bash
# Check NCCL communication
docker exec axolotl-usf-master env | grep NCCL

# Test network connectivity
docker exec axolotl-usf-master ping worker1-ip
```

### Monitor Training Progress

```bash
# View logs from all nodes
# On master
docker logs -f axolotl-usf-master

# On workers
docker logs -f <worker_container_name>
```

## 🎯 Different Configurations

### 4 Nodes × 1 GPU = 4 Total GPUs

```bash
# Master (Node 0)
WORLD_SIZE=4 GPUS_PER_NODE=1 MASTER_ADDR=192.168.1.100 \
  docker-compose -f integration/docker/docker-compose.multi-node.yml up master

# Workers (Nodes 1-3)
WORLD_SIZE=4 GPUS_PER_NODE=1 MASTER_ADDR=192.168.1.100 NODE_RANK=1 \
  docker-compose -f integration/docker/docker-compose.multi-node.yml up worker

WORLD_SIZE=4 GPUS_PER_NODE=1 MASTER_ADDR=192.168.1.100 NODE_RANK=2 \
  docker-compose -f integration/docker/docker-compose.multi-node.yml up worker

WORLD_SIZE=4 GPUS_PER_NODE=1 MASTER_ADDR=192.168.1.100 NODE_RANK=3 \
  docker-compose -f integration/docker/docker-compose.multi-node.yml up worker
```

### 4 Nodes × 4 GPUs = 16 Total GPUs

```bash
# Master (Node 0)
WORLD_SIZE=16 GPUS_PER_NODE=4 MASTER_ADDR=192.168.1.100 \
  docker-compose -f integration/docker/docker-compose.multi-node.yml up master

# Workers (Nodes 1-3)
WORLD_SIZE=16 GPUS_PER_NODE=4 MASTER_ADDR=192.168.1.100 NODE_RANK=1 \
  docker-compose -f integration/docker/docker-compose.multi-node.yml up worker

WORLD_SIZE=16 GPUS_PER_NODE=4 MASTER_ADDR=192.168.1.100 NODE_RANK=2 \
  docker-compose -f integration/docker/docker-compose.multi-node.yml up worker

WORLD_SIZE=16 GPUS_PER_NODE=4 MASTER_ADDR=192.168.1.100 NODE_RANK=3 \
  docker-compose -f integration/docker/docker-compose.multi-node.yml up worker
```

### 4 Nodes × 8 GPUs = 32 Total GPUs (DGX nodes)

```bash
# Master (Node 0)
WORLD_SIZE=32 GPUS_PER_NODE=8 MASTER_ADDR=192.168.1.100 \
  docker-compose -f integration/docker/docker-compose.multi-node.yml up master

# Workers (Nodes 1-3)
WORLD_SIZE=32 GPUS_PER_NODE=8 MASTER_ADDR=192.168.1.100 NODE_RANK=1 \
  docker-compose -f integration/docker/docker-compose.multi-node.yml up worker

WORLD_SIZE=32 GPUS_PER_NODE=8 MASTER_ADDR=192.168.1.100 NODE_RANK=2 \
  docker-compose -f integration/docker/docker-compose.multi-node.yml up worker

WORLD_SIZE=32 GPUS_PER_NODE=8 MASTER_ADDR=192.168.1.100 NODE_RANK=3 \
  docker-compose -f integration/docker/docker-compose.multi-node.yml up worker
```

## 🔧 Advanced Configuration

### Using .env Files

Create `.env` file on each node:

**Master node (.env):**
```bash
WORLD_SIZE=8
GPUS_PER_NODE=2
MASTER_ADDR=192.168.1.100
MASTER_PORT=29500
NODE_RANK=0
```

**Worker nodes (.env):**
```bash
WORLD_SIZE=8
GPUS_PER_NODE=2
MASTER_ADDR=192.168.1.100
MASTER_PORT=29500
NODE_RANK=1  # Change to 2, 3 for other workers
```

Then simply run:
```bash
docker-compose -f integration/docker/docker-compose.multi-node.yml up master  # or worker
```

### NCCL Optimization

For InfiniBand networks:
```bash
WORLD_SIZE=8 GPUS_PER_NODE=2 MASTER_ADDR=192.168.1.100 \
  NCCL_IB_DISABLE=0 NCCL_SOCKET_IFNAME=ib0 \
  docker-compose -f integration/docker/docker-compose.multi-node.yml up master
```

For Ethernet networks (default):
```bash
WORLD_SIZE=8 GPUS_PER_NODE=2 MASTER_ADDR=192.168.1.100 \
  NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=eth0 \
  docker-compose -f integration/docker/docker-compose.multi-node.yml up master
```

### Debug Mode

Enable NCCL debugging:
```bash
WORLD_SIZE=8 GPUS_PER_NODE=2 MASTER_ADDR=192.168.1.100 \
  NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=ALL \
  docker-compose -f integration/docker/docker-compose.multi-node.yml up master
```

## ⚠️ Common Issues

### Issue: Nodes can't communicate

**Solution:**
```bash
# Check firewall on all nodes
sudo ufw status
sudo ufw allow 29500/tcp

# Test connectivity
ping 192.168.1.100  # from worker nodes
telnet 192.168.1.100 29500  # test port
```

### Issue: NCCL timeout

**Solution:**
```bash
# Increase timeout (default 1800 seconds)
NCCL_TIMEOUT=3600 docker-compose -f integration/docker/docker-compose.multi-node.yml up master
```

### Issue: GPU not detected on workers

**Solution:**
```bash
# Verify NVIDIA runtime on all nodes
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi

# Check docker daemon.json on all nodes
cat /etc/docker/daemon.json
```

## 📊 Monitoring

### View GPU Utilization Across All Nodes

Create a monitoring script:
```bash
#!/bin/bash
# monitor-gpus.sh

echo "=== Node 0 (Master) ==="
ssh user@192.168.1.100 "docker exec axolotl-usf-master nvidia-smi"

echo "=== Node 1 (Worker) ==="
ssh user@192.168.1.101 "docker exec <worker-container> nvidia-smi"

echo "=== Node 2 (Worker) ==="
ssh user@192.168.1.102 "docker exec <worker-container> nvidia-smi"

echo "=== Node 3 (Worker) ==="
ssh user@192.168.1.103 "docker exec <worker-container> nvidia-smi"
```

## 🎓 Key Takeaways

1. **WORLD_SIZE** = Total number of GPUs across all nodes
2. **GPUS_PER_NODE** = Number of GPUs on each node (should be same on all nodes)
3. **NODE_RANK** = Unique identifier for each node (0 for master, 1,2,3... for workers)
4. **MASTER_ADDR** = IP address of the master node (same on all nodes)
5. **Start order**: Always start master first, then workers

## 📚 Additional Resources

- [PyTorch Distributed Tutorial](https://pytorch.org/tutorials/intermediate/dist_tuto.html)
- [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/)
- [Docker Multi-Node Setup](../README.md)

---

**Ready to scale!** 🚀
