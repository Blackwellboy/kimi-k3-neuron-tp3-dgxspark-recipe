# APPLY.md — SparkInfer K3 multi-Spark patch chain

**Base commit (required):** `7a9b77a043596157d74e4af376cf9f29f68ce368`  
**Recommended tag:** `sparkinfer-tp3-phase3-loadready-fix` (tracks current main tip; always am **0001–0009**)

## Apply (fresh tree)

```bash
git clone https://github.com/gittensor-ai-lab/sparkinfer-k3.git
cd sparkinfer-k3
git fetch origin 7a9b77a043596157d74e4af376cf9f29f68ce368
git checkout -B k3-tp3-phase3 7a9b77a043596157d74e4af376cf9f29f68ce368

REPO=https://github.com/vcruz305/kimi-k3-neuron-tp3-dgxspark-recipe
git clone --depth 1 --branch sparkinfer-tp3-phase3-loadready-fix "$REPO" /tmp/k3-recipe

for p in \
  /tmp/k3-recipe/patches/sparkinfer/0001-tp-plan-K3-all-expert-FFN-width-shards-for-TP3.patch \
  /tmp/k3-recipe/patches/sparkinfer/0002-wire-tp3-all-expert-width-init-ffn-prefill.patch \
  /tmp/k3-recipe/patches/sparkinfer/next/0003-tp-three-host-rank-bootstrap-protocol.patch \
  /tmp/k3-recipe/patches/sparkinfer/next/0004-tp-distributed-rank-transport-and-tp3-tp4-plans.patch \
  /tmp/k3-recipe/patches/sparkinfer/next/0005-tp-rank-local-plan-and-nccl-microbench.patch \
  /tmp/k3-recipe/patches/sparkinfer/next/0006-tp-rank-local-gguf-load-api-and-moe-budget.patch \
  /tmp/k3-recipe/patches/sparkinfer/next/0007-tp-distributed-eager-forward-and-dist-generate.patch \
  /tmp/k3-recipe/patches/sparkinfer/next/0008-tp-fix-rank0-loadready-oneshot-and-load-before-ready.patch \
  /tmp/k3-recipe/patches/sparkinfer/next/0009-tp-f16-token-embd-output-and-dist-embed.patch
do
  git am "$p"
done
```

Build (NCCL required for real multi-GPU/Spark generate):

```bash
cmake -S runtime -B build -DSPARKINFER_TP=ON
cmake --build build -j"$(nproc)" --target kimi_k3_dist_generate \
  tp_rank_local_loader_cpu_test tp_dist_generate_protocol_cpu_test
# optional kernel smokes (if built): kimi_k3_row_dequant_test kimi_k3_proj_gemv_test
```

## Patch series (9 commits)

| # | Purpose |
|---|---|
| 0001 | AllExpertsFfnWidth 512/512/512 planner |
| 0002 | wire + NCCL prefill tile auto-disable |
| 0003 | multi-host protocol |
| 0004 | TCP transport + TP3/TP4 |
| 0005 | rank-local plan + NCCL microbench |
| 0006 | rank-local GGUF load + MoE budget |
| 0007 | dist forward + `kimi_k3_dist_generate` |
| 0008 | rank0 LoadReady one-shot + load-before-ready |
| **0009** | **F16 token_embd/output kernels + dist `embed_token` type 1** |

### 0008 (LoadReady) — verified on 3×GB10
External qual: 3/3 load ~32 min, no duplicate LoadReady, NCCL green.

### 0009 (F16 embed) — why
After 0008, all ranks hit `forward: embed_token failed` on first token.
Neuron GGUF `token_embd.weight` is **GGML F16 (type 1)**; dist embed only allowed F32/Q8_0.
0009 ports the known F16 boundary kernels (row dequant + dense proj) and sizes F16 rows
in dist/TP/single-GPU embed. Rank0 LM head uses `k3_proj_f32` which accepts wtype 1.

## Phase 3 usage (3 Sparks)

```bash
./kimi_k3_dist_generate --rank 0 --world 3 --listen 0.0.0.0:29500 \
  --model /data/k3-neuron-iq1s-00001-of-00009.gguf \
  --prompt-ids 1,2,3 --n-predict 8 --max-ctx 8192

./kimi_k3_dist_generate --rank 1 --world 3 --coord HOST:29500 \
  --model /data/k3-neuron-iq1s-00001-of-00009.gguf --max-ctx 8192
./kimi_k3_dist_generate --rank 2 --world 3 --coord HOST:29500 \
  --model /data/k3-neuron-iq1s-00001-of-00009.gguf --max-ctx 8192
```

Env recommended: `SPARKINFER_K3_MOE_WEPS=0 SPARKINFER_K3_GRAPH=0 NCCL_NVLS_ENABLE=0` + auto GID.

## Verified

| Gate | Status |
|---|---|
| git am 0001–0009 | format-patch chain |
| External 0008 3×GB10 | load 3/3 PASS; LoadReady fixed |
| External forward pre-0009 | FAIL embed_token (F16) |
| Multi-Spark generate after 0009 | **pending retest** |

## Still experimental
No multi-Spark tok/s claim until short generate + parity receipts after 0009.
