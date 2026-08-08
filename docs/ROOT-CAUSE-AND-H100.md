# H100 session + root-cause of earlier GPU failures (2026-08-07)

## Inventory (at session time)

| Offer | Rate | Fit |
|---|---:|---|
| 1× H100 80GB | $1.75–$2.24/hr | Health / template only |
| **2× H100 80GB** | **$3.50/hr** | Same-host multi-rank NCCL ✅ |
| 8× H100 80GB | $14+/hr | Over prior $12/hr cap |
| H100 VRAM | **80 GB** | **Cannot** hold TP3 ~113 GiB/rank full model |

Template used: **Stable PyTorch 2.12 CUDA 13.0 DIND**  
(`primarchleman/lium-pytorch:2.12.0-cu130-dind-ubuntu24.04-docker27`)  
Includes **CUDA toolkit 13.0 + nvcc**, torch 2.12 cu130, DIND.

---

## What we ran

### A) Three separate 1×H100 pods (~$6.23/hr total) — multi-host attempt
- Austin HBM3 + 2× Gardner PCIe  
- **Health gate: all three PASS** (matmul, ECC 0, free ~78 GiB)  
- **Cross-pod TCP:** only **SSH ports** reachable between pods  
- Published **8000/8888 mappings time out** peer-to-peer (even Gardner↔Gardner)  
- ⇒ **Multi-host NCCL cannot form a data plane on Lium without extra tunnels/VPN**  
- Pods deleted after diagnosis  

### B) Same-node **2×H100** Atlanta @ **$3.50/hr** (1h cap, released early)

| Check | Result |
|---|---|
| GPU0/1 health matmul | **PASS**, ECC 0 |
| Torch 2-rank NCCL | **PASS** — 7168 f32 ~**24.6 µs**; 10752 f32 ~**32.0 µs** (~4.5–5.9 ms @ 185 red.) |
| C++ `ncclCommInitRank` microbench | **PASS** — ~**8.1–8.2 µs**; ~**1.52 ms/token** @ 185 red. |
| nvcc 13.0 | Present at `/usr/local/cuda/bin/nvcc` |

Logs: `h100x2-results.txt`

---

## Root cause: why the earlier 3×H200 run failed

Several **independent** failures stacked. Not one mystery bug.

### 1. Bad node / GPU health (primary)
- `nvidia-smi` showed GPUs present, but **torch could only use GPU1**
- GPU0/2: `cudaErrorDevicesUnavailable`
- Elevated **uncorrectable ECC** on at least GPU2  
- `nvidia-smi --gpu-reset` failed (“Unable to disable persistence mode”)  
- **Fix:** hard health gate before any long job — matmul on **every** index; delete pod if any fail  

### 2. Wrong first template for native builds
- First H200 used a thin PyTorch image: **no nvcc on PATH**, incomplete pip-only CUDA headers (`crt/host_config.h` missing / recursive shim)  
- **Stable PyTorch 2.12 CUDA 13.0 DIND** fixes this: full toolkit + nvcc  
- Note: health script reported `nvcc: missing` until PATH includes `/usr/local/cuda/bin`  

### 3. `CUDA_VISIBLE_DEVICES` + broken GPUs
- One-device-per-process is correct for rank simulation, but if GPU0 is dead, rank0 dies immediately and the pack hangs on TCPStore  
- **Fix:** health gate first; bind by healthy ordinals  

### 4. Multi-pod Lium networking (H100 three-singles lesson)
- Inter-pod **SSH works**  
- Inter-pod **app ports (8000/8888) do not** (timeout)  
- TCPStore/NCCL bootstrap over “public mapped non-SSH ports” **fails** between rentals  
- **Implication for 3 Sparks:** real Sparks need **cluster fabric (RoCE/IB)** or explicitly opened data ports — Lium multi-pod is **not** a drop-in Spark network stand-in  
- **Implication for bring-up:** prefer **one multi-GPU node** for InitRank proof; use multi-pod only with tunnels/VPN if required  

### 5. Not the model / not TP3 geometry
- Failures happened **before** any GGUF load  
- TP3 512-wide plan was never reached on the bad H200  

---

## Budget (this H100 push)

| | |
|---|---|
| Approx additional spend | small (minutes on 3×1 + short 2×) |
| End balance | see live `/users/me` after delete (~$354.x) |
| Pods left | **0** |

---

## What this means for 3-Spark production

| Item | Status |
|---|---|
| Multi-process `ncclCommInitRank` | **Proven** (H100 2-rank native C++ + torch) |
| Same-host collective latency | **~8 µs** H100 native; ~25–32 µs torch path |
| Multi-host on Lium | **Blocked by port filter** (SSH only) |
| Full TP3 load on H100 80GB | **No** — need H200-class or Sparks (~113 GiB/rank) |
| Template for future pods | **Stable PyTorch 2.12 CUDA 13.0 DIND** |
| Health gate | **Mandatory** before spend |

## Next

1. When a **healthy 3×H200 ≤ $12/hr** appears: health-gate all 3 → NCCL 3-rank → rank-local load.  
2. On real Sparks: use fabric, not Lium’s SSH-only mesh.  
3. Keep using templated CUDA-13 DIND images so nvcc/NCCL builds don’t thrash.
