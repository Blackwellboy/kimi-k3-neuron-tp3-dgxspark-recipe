# Kimi-K3 Neuron — 3-Spark TP3 bring-up package

**Status: preview / bring-up — not a drop-in multi-Spark production server.**

End-to-end **load → generate → finish** on **3× NVIDIA DGX Spark** is working with patches
**0001–0012**. Decode speed is in the stretch band for single-stream exact TP3.

| | |
|---|---|
| **This recipe** | [`github.com/vcruz305/kimi-k3-neuron-tp3-dgxspark-recipe`](https://github.com/vcruz305/kimi-k3-neuron-tp3-dgxspark-recipe) |
| **Model (GGUF)** | [`huggingface.co/vcruz305/Kimi-K3-Neuron-IQ1S-GGUF`](https://huggingface.co/vcruz305/Kimi-K3-Neuron-IQ1S-GGUF) (manual-gated, ~330 GB / 307.49 GiB) |
| Upstream SparkInfer base | `gittensor-ai-lab/sparkinfer-k3` @ `7a9b77a043596157d74e4af376cf9f29f68ce368` |
| Geometry | **AllExpertsFfnWidth TP3** — 896 experts/rank, FFN 512/512/512, `ncclCommInitRank` |
| Patch tip | **0001–0012** (see [`APPLY.md`](APPLY.md)) |

---

## Benchmarks on 3× DGX Spark (measured)

Hardware: 3× DGX Spark (GB10). Model on **local NVMe**.  
`max_ctx=8192`, `SPARKINFER_K3_MOE_WEPS=0`, `SPARKINFER_K3_GRAPH=0`, `NCCL_NVLS_ENABLE=0`.  
Binary: `kimi_k3_dist_generate` after **0012**.

| Run | decode tok/s | tokens | vs RPC 2.85 | notes |
|-----|-------------:|-------:|------------:|-------|
| Clean finish | **5.67** | 32 | **1.99×** | first full e2e |
| Longer window | **6.21** | 128 | **2.18×** | more stable decode |
| Historical llama.cpp RPC layer-split | **2.85** | — | 1.0× | production path today on many 3-Spark setups |
| Same-host TP3 (reference only) | ~42 | — | — | **3×H200 one box** — not Sparks |

Forecast ([`docs/SPARK-TP3-PERFORMANCE-FORECAST.md`](docs/SPARK-TP3-PERFORMANCE-FORECAST.md)):
likely **4.5–6.0**, stretch **6.5–7.5**. **6.21 is stretch-band.** Peer multi-prompt hygiene
has reported ~**6.5–7.2**. **~10 tok/s is not available from same-recipe polish** (needs
batching / different split / more nodes / other engine).

**Context ceiling on 3 Sparks:** ~**113 GiB/rank** weights → practical **~8K** ctx. 12K/16K
loads have OOM’d. Not a 1M-context path.

**Quality:** speed runs with raw `--prompt-ids` can look degenerate. For quality, tokenize with
the HF chat template (`k3_chat_template.jinja`) first.

---

## How to run on 3 Sparks

### 0. Prerequisites

- 3 Sparks with SSH, working RoCE/fabric (or at least TCP + NCCL), CUDA 13 / driver stack aligned
- ~330 GB **local NVMe** free per node for the GGUF (avoid NFS for weights)
- HF access to the gated model + `hf auth login`

### 1. Download the model (each Spark, local disk)

```bash
pip install -U "huggingface_hub[hf_xet]"
hf auth login
export HF_XET_HIGH_PERFORMANCE=1
hf download vcruz305/Kimi-K3-Neuron-IQ1S-GGUF \
  --local-dir $HOME/models/kimi-k3-neuron-iq1s \
  --include "*.gguf" --include "k3_chat_template.jinja"
```

### 2. Build SparkInfer + apply this recipe (0001–0012)

See **[`APPLY.md`](APPLY.md)** for the full `git am` list. Short form:

```bash
git clone https://github.com/gittensor-ai-lab/sparkinfer-k3.git && cd sparkinfer-k3
git checkout -B k3-tp3 7a9b77a043596157d74e4af376cf9f29f68ce368

git clone --depth 1 \
  https://github.com/vcruz305/kimi-k3-neuron-tp3-dgxspark-recipe /tmp/k3-recipe

# apply 0001–0012 exactly as in APPLY.md, then:
cmake -S runtime -B build -DSPARKINFER_TP=ON
cmake --build build -j"$(nproc)" --target kimi_k3_dist_generate \
  tp_rank_local_loader_cpu_test tp_dist_generate_protocol_cpu_test

mkdir -p $HOME/k3-tp3/dist
cp -f build/kimi_k3_dist_generate build/libsparkinfer_runtime.so $HOME/k3-tp3/dist/
# copy the same dist/ tree to all three Sparks
```

### 3. Topology

```text
Spark A  rank 0  CUDA0   listen 0.0.0.0:29500   coordinator + sampler + full LM head
Spark B  rank 1  CUDA0   --coord A_FABRIC_IP:29500   worker
Spark C  rank 2  CUDA0   --coord A_FABRIC_IP:29500   worker
         └── NCCL ncclCommInitRank over fabric ──┘
         └── TCP control plane (Hello → NCCL id → Load → Token → Finish) ──┘
```

Use Spark A’s **fabric** IP (e.g. `10.10.10.x`), not a random public hostname, for `--coord`.

### 4. Launch (rank0 first)

```bash
export LD_LIBRARY_PATH=$HOME/k3-tp3/dist:$LD_LIBRARY_PATH
export SPARKINFER_K3_MOE_WEPS=0
export SPARKINFER_K3_GRAPH=0
export NCCL_NVLS_ENABLE=0
export CUDA_VISIBLE_DEVICES=0
MODEL=$HOME/models/kimi-k3-neuron-iq1s/k3-neuron-iq1s-00001-of-00009.gguf
BIN=$HOME/k3-tp3/dist/kimi_k3_dist_generate
COORD=10.10.10.2   # Spark A fabric IP

# --- Spark A (rank 0) ---
$BIN --rank 0 --world 3 --listen 0.0.0.0:29500 \
  --model "$MODEL" --prompt-ids 1,2,3 --n-predict 128 --max-ctx 8192 \
  2> r0.stderr | tee r0.stdout

# --- Spark B (rank 1) ---
$BIN --rank 1 --world 3 --coord ${COORD}:29500 \
  --model "$MODEL" --max-ctx 8192 2> r1.stderr | tee r1.stdout

# --- Spark C (rank 2) ---
$BIN --rank 2 --world 3 --coord ${COORD}:29500 \
  --model "$MODEL" --max-ctx 8192 2> r2.stderr | tee r2.stdout
```

Success markers on rank0:

- `rank 0 loaded ok`
- `decode_tok_s=...`
- `OK finished clean`

First load is long (~30–60+ minutes for ~88 GiB MoE slice/rank onto GPU).

### 5. Optional health / NCCL before long loads

```bash
# per GPU
python scripts/gpu_health_gate.py health

# 3-rank microbench helpers live under scripts/ (see docs)
```

---

## Documents

| File | Audience |
|---|---|
| **[`APPLY.md`](APPLY.md)** | Clean `git am` chain **0001–0012** |
| **[`SPARK-AGENT.md`](SPARK-AGENT.md)** | LLM agents operating Sparks |
| [`THREE-SPARK-TP3-RECIPE.md`](THREE-SPARK-TP3-RECIPE.md) | Geometry + ladder |
| [`docs/SPARK-TP3-PERFORMANCE-FORECAST.md`](docs/SPARK-TP3-PERFORMANCE-FORECAST.md) | Forecast + stop rules |
| [`docs/OPERATOR-3-AND-4-SPARK.md`](docs/OPERATOR-3-AND-4-SPARK.md) | 3- and 4-Spark geometry |
| [`docs/TP3-EXPLAINED-AND-FIXED.md`](docs/TP3-EXPLAINED-AND-FIXED.md) | TP vs layer-split |
| [`evidence/`](evidence/) | NCCL / wire-up receipts |
| [`patches/sparkinfer/`](patches/sparkinfer/) | Frozen patches |

---

## Hard non-claims

Do **not** advertise this package as:

- install-and-run multi-Spark **production** inference;
- **40+ tok/s on three Sparks** (that number is **same-host 3×H200**);
- bit-identical to llama.cpp without parity gates;
- **1M context on 3 Sparks** with full resident TP3;
- support for `matched-byte-v2` on SparkInfer (llama.cpp path for now).

---

## License / provenance

- Patches target **SparkInfer-K3** (upstream license + NOTICE apply when merging).
- Recipe docs and agent brief: MIT unless noted.
- Model weights: see [HF card](https://huggingface.co/vcruz305/Kimi-K3-Neuron-IQ1S-GGUF) (modified-MIT / gated).
- Preserve upstream attribution; do not strip NOTICE from kernel-derived paths.
