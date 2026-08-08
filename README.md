# Kimi-K3 Neuron — 3-Spark TP3 bring-up package (preview)

**Status: preview / bring-up — not a drop-in multi-Spark production server.**

**Phase 3 + 0009:** distributed generate path + rank0 LoadReady one-shot/load-before-ready fix. Tag `sparkinfer-tp3-phase3-loadready-fix` (apply 0001–0009).

This repository packages the frozen geometry, control-plane patches, collective
proofs, and **LLM agent instructions** for developers who operate NVIDIA DGX
Spark fleets and want to continue Kimi-K3 Neuron IQ1_S tensor-parallel work on
**three Sparks**.

| | |
|---|---|
| Model artifact | [`vcruz305/Kimi-K3-Neuron-IQ1S-GGUF`](https://huggingface.co/vcruz305/Kimi-K3-Neuron-IQ1S-GGUF) (manual-gated, ~330 GB) |
| Upstream SparkInfer base | `gittensor-ai-lab/sparkinfer-k3` @ `7a9b77a043596157d74e4af376cf9f29f68ce368` |
| Production runtime today | **llama.cpp** RPC layer-split (~2.85 tok/s on 3 Sparks historically) |
| Same-host TP proof | SparkInfer AllExpertsFfnWidth TP3 ~**42 tok/s** on 3×H200 |
| Multi-Spark full generate | **Not shipped yet** — see gaps below |

## Documents

| File | Audience |
|---|---|
| **[`APPLY.md`](APPLY.md)** | **Clean git am chain (Phase 0 — use this)** |
| **[`SPARK-AGENT.md`](SPARK-AGENT.md)** | **LLM agents that control Sparks for developers** (primary runbook) |
| [`THREE-SPARK-TP3-RECIPE.md`](THREE-SPARK-TP3-RECIPE.md) | Human / agent shared recipe |
| [`docs/TP3-EXPLAINED-AND-FIXED.md`](docs/TP3-EXPLAINED-AND-FIXED.md) | TP vs layer-split, tile/NCCL pitfall |
| [`docs/SPARK-TP3-PERFORMANCE-FORECAST.md`](docs/SPARK-TP3-PERFORMANCE-FORECAST.md) | 4.5–6.0 tok/s forecast + stop rules |
| [`docs/OPERATOR-3-AND-4-SPARK.md`](docs/OPERATOR-3-AND-4-SPARK.md) | 3- and 4-Spark geometry |
| [`docs/ROOT-CAUSE-AND-H100.md`](docs/ROOT-CAUSE-AND-H100.md) | Ops failures (bad GPUs, templates, multi-pod ports) |
| [`evidence/`](evidence/) | Measured receipts (NCCL, wire-up) |
| [`patches/sparkinfer/`](patches/sparkinfer/) | Frozen `git am` chain |
| [`scripts/`](scripts/) | Health gate + NCCL microbenches |

## Hard non-claims

Do **not** advertise this package as:

- install-and-run multi-Spark production inference;
- 40+ tok/s on three Sparks (that number is **same-host 3×H200**);
- bit-identical to llama.cpp without parity gates;
- support for `matched-byte-v2` on SparkInfer (llama.cpp only for now).

## Quick layout

```text
patches/sparkinfer/
  0001-tp-plan-...patch
  0002-wire-tp3-...patch
  next/
    0002-tp-specify-three-host-rank-bootstrap-protocol.patch
    0003-tp-distributed-rank-transport-and-tp3-tp4-plans.patch
    0004-tp-rank-local-plan-and-nccl-microbench.patch
scripts/
  gpu_health_gate.py
  torch_nccl_rank.py
  k3_dist_nccl_microbench.cpp
```

## Apply chain (SparkInfer tree)

```bash
git clone https://github.com/gittensor-ai-lab/sparkinfer-k3.git
cd sparkinfer-k3
git checkout 7a9b77a043596157d74e4af376cf9f29f68ce368
git am /path/to/this/repo/patches/sparkinfer/0001-*.patch
git am /path/to/this/repo/patches/sparkinfer/0002-*.patch
git am /path/to/this/repo/patches/sparkinfer/next/0002-*.patch
git am /path/to/this/repo/patches/sparkinfer/next/0003-*.patch
git am /path/to/this/repo/patches/sparkinfer/next/0004-*.patch
```

Verify hashes with `SHA256SUMS`.

## License / provenance

- Patches target **SparkInfer-K3** (upstream license + NOTICE apply when merging).
- Recipe docs and agent brief: MIT unless noted.
- Model weights: see Hugging Face card (modified-MIT / gated access).
- Preserve upstream attribution; do not strip NOTICE from kernel-derived paths.

## Maintainers

Prepared for the Victor Cruz / Cruzzin Spark developer workflow.  
Questions: open an issue on this repo.
