# Distributed ranks for 3- and 4-Spark users

Status: **control-plane + geometry + collective API milestone**.  
Not yet a drop-in full-model multi-Spark server. Production remaining work is listed below.

## Who this is for

| Fleet | Geometry | Memory plan (release 330 GB IQ1_S) |
|---|---|---|
| **3 Sparks** | `AllExpertsFfnWidth` — all 896 experts every rank, FFN **512/512/512** | ~112.85 GiB/rank (tight on 128 GB) |
| **4 Sparks** | `ExpertFfn2D` eg=2/fs=2 — **448** experts/group × FFN **768/768** | ~84.06 GiB/rank (preferred headroom) |

Same process model for both: **one process, one local CUDA device 0, one global rank per Spark**.

## What landed in this milestone (`0003`)

1. **Protocol** extended for `world_size ∈ {3,4}` with fail-closed validation.
2. **TCP framed transport** (`rank_transport.h` / `rank_transport_tcp.cpp`):
   - 522-byte frame hard limit; reject oversize payload **before** body allocation
   - connection bind per rank, timeouts, coordinator + worker helpers
3. **Rank collective API** (`rank_collective.h` / `.cpp`):
   - `make_nccl_rank_collective` → `ncclCommInitRank` when `SPARKINFER_TP_NCCL`
   - `make_host_mock_rank_collective` for CPU multi-thread tests
   - leaves existing same-host `ncclCommInitAll` path untouched
4. **CPU tests** (`tp_dist_rank_cpu_test`):
   - TP3 + TP4 full protocol dry-run (Hello→NCCL id→Load→Tokens→Finish)
   - host-mock all-reduce for world 3 and 4
   - TCP round-trip + oversize rejection
5. Prior protocol suite still green (**560/560** after TP4 accept).

## Live evidence (2026-08-07)

- **3-process NCCL InitRank-style all-reduce PASS** on world=3 (8×5090 pod, $4.80/hr).
  K3-sized payloads 28,672 / 43,008 B: correct sums, ~**41.5 µs**/call (~**7.7 ms**/token at 185 reductions).
  Receipt: `pod-nccl3-20260807/RECEIPT.md`.
- A 3×H200 @ $8.25/hr was tried for full TP3 residency and **deleted** after GPUs 0/2 were unavailable.
- Session spend ~**$1.55**; no idle pods left running.

## What is **not** production yet

| Missing | Why it blocks Spark users |
|---|---|
| Rank-local GGUF loader (slice-only residency) | Full 330 GB per rank will OOM / page on 128 GB Sparks |
| `KimiK3DistributedRank` eager forward | No tokens without model path |
| Multi-host NCCL microbench on real RoCE | Latency unknown; forecast only |
| `kimi_k3_dist_generate` operator binary | CLI not shipped as end-to-end generator |
| Long-context parity (KLD/top-1/PPL) vs llama.cpp | Speed claims without parity are invalid |
| CUDA graph / WEPS | Stay off until eager parity |

## Operator apply chain (developers)

```bash
git checkout 7a9b77a043596157d74e4af376cf9f29f68ce368
git am 0001-tp-plan-K3-all-expert-FFN-width-shards-for-TP3.patch
git am 0002-wire-tp3-all-expert-width-init-ffn-prefill.patch   # if using wire-up
git am next/0003-tp-three-host-rank-bootstrap-protocol.patch
git am next/0004-tp-distributed-rank-transport-and-tp3-tp4-plans.patch
```

CPU smoke (no GPU):

```bash
# standalone
g++ -std=c++17 -O1 -pthread -Iruntime/include \
  runtime/src/tp/rank_protocol.cpp \
  runtime/src/tp/rank_transport_tcp.cpp \
  runtime/src/tp/rank_collective.cpp \
  runtime/src/tp/shard.cpp \
  runtime/src/tp/weight_plan.cpp \
  runtime/tests/tp_dist_rank_cpu_test.cpp \
  -o tp_dist_rank_cpu_test
./tp_dist_rank_cpu_test
```

## Qualification ladder (unchanged; must pass before any tok/s claim)

1. CPU protocol + TCP tests (this milestone) ✅  
2. Three/four localhost processes with real TCP bootstrap + host mock reduce  
3. NCCL f32 all-reduce microbench across Sparks (RoCE, no silent TCP fallback)  
4. Rank-local 8K load, zero swap, ≥8 GiB `MemAvailable`  
5. One-token greedy top-1 vs llama.cpp / same-host TP  
6. Multi-token + long prompt KLD/top-1/RMS  
7. Only then decode tok/s vs llama RPC ~2.85 baseline  

Forecast after full port: **4.5–6.0 tok/s likely** on 3 Sparks (not 40+).  
Stop rule: abandon single-stream TP3 if finished RDMA+graphs ≤3.5 tok/s.

## Recommended product messaging (honest)

> Multi-Spark distributed TP for Kimi-K3 Neuron is **under active bring-up**.  
> Same-host SparkInfer TP3 is proven (~42 tok/s on 3×H200).  
> Cross-Spark control plane, TP3/TP4 geometry, and NCCL rank API are merged as  
> intermediate artifacts. Full-model multi-Spark generation is the next release.

Do **not** tell 3/4-Spark owners this is install-and-run production speedware yet.
