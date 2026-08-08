# Three-DGX-Spark TP3 performance forecast

Date: 2026-08-07  
Artifact: 330.2 GB / 307.49 GiB public K3-Neuron GGUF  
Baseline: operator-measured llama.cpp RPC layer split, 2.85 tok/s decode

These are engineering ranges, not measurements. They assume a correct rank-local,
full-expert (`WEPS=0`) distributed TP3 implementation, real 8K context, no paging,
and numerical qualification against llama.cpp.

| Exact TP3 single stream | Decode forecast |
|---|---:|
| lower | 3.3-4.0 tok/s |
| likely | **4.5-6.0 tok/s** |
| stretch | 6.5-7.5 tok/s |

Unmodified SparkInfer produces **0 tok/s** on three separate Sparks: its TP backend is
single-process/local-device `ncclCommInitAll`, its public generator is layer-split, and
its current TP executor rejects `896 % 3 != 0`.

## Why the range is plausible

- The measured serial RPC rate is 2.85 tok/s, or 350.88 ms/token. With three equal
  serial stages, ideal overlap would be 116.96 ms/token or 8.55 tok/s. That is a
  scheduling ceiling, not a forecast.
- The GGUF contains 265.893 GB of routed-expert bytes. Selecting 16 of 896 implies
  about 4.748 GB of routed weights read per token. After excluding the embedding-row
  lookup while retaining the streamed output head, total active weight traffic is
  approximately 66.7 GB/token, or 22.2 GB/rank under ideal width TP3.
- Each Spark advertises 273 GB/s memory bandwidth. The loose three-rank bandwidth
  roof is about 12.3 tok/s, but the measured llama stage already realizes only around
  70% of that simple roof. Narrow 512-wide kernels, replicated reads, launch overhead,
  and synchronization lower the real ceiling.
- The corrected graph has 185 dependency-ordered reductions/token. Logical payload is
  only about 6.62 MB/token; a three-rank ring moves roughly 8.83 MB/rank, around
  0.35 ms at 200 Gb/s. Payload bandwidth is therefore not the primary network cost.
  Collective latency is: 50/100/150/200/300 microseconds per call implies roughly
  9.25/18.5/27.75/37.0/55.5 ms/token before other rendezvous costs.

A same-host three-H200 microprobe now supplies a measured lower bound. With NVLS
disabled, blocking 28,672/43,008-byte NCCL reductions measured 32.543/31.445 us p50
and 48.805/41.132 us p95. Multiplying by 185 gives roughly 5.8-6.0 ms/token at p50
and 7.6-9.0 ms/token at p95. This does not predict Spark RoCE latency, but it confirms
that per-call latency is measurable even when payload serialization is negligible.
Full receipt: `NCCL-LATENCY-RECEIPT.md`.

## Principal blocker: memory headroom

The real plan is 112.85 GiB/rank. A 128 GB decimal Spark exposes at most about
119.21 GiB before system reservations, leaving only 6.36 GiB for the OS, CUDA/NCCL,
KV/state, scratch, and file-cache pressure. That is not a safe executable plan yet.

The first implementation must save roughly 6-10 GiB/rank through rank-local/no-copy
loading, head/embedding banding, reduced replicas, and bounded scratch—or use TP4.
TP4 plans at 84.06 GiB/rank and has about 35.15 GiB nominal headroom, but four separate
Sparks still require the distributed runtime and NVIDIA's supported switch topology.

## Minimal qualification matrix

1. Three-rank NCCL microbench at 28,672 and 43,008 FP32 bytes, 10,000 calls. Record
   p50/p95 latency, chosen NIC, GPUDirect/RDMA evidence, and socket fallback.
2. Rank-local 8K load. Require zero swap/UVM migration and at least 8 GiB steady
   `MemAvailable`; stop and fix residency if either condition fails.
3. Tiny full-K3 TP1 versus TP3 at positions 128 and 8K. Require deterministic greedy
   top-1 identity and logit NMSE <= 1e-4 before speed credit.
4. Full release, `WEPS=0`, fixed 64-token output, five warmed repetitions: llama RPC
   versus TP3 at short context, 8K, and naturally ingested 32K. Record median/p95 ITL,
   prefill, per-rank memory, and collective time.
5. Only after exact qualification, test 32K with `WEPS=0.08`, record `WEPS_SKIPS`,
   KLD/top-1/RMS/PPL, and label it approximate.
6. Test concurrency 3 separately for aggregate throughput; do not present it as
   single-user latency.

## Stopping rule

Stop immediately on paging/UVM migration or parity failure. After rank-parallel issue,
CUDA graphs, and verified RDMA, stop the TP3 single-stream branch if exact median decode
remains <=3.5 tok/s (<=23% over the 2.85 baseline) or p95 latency regresses badly.
Continue cautiously at 3.5-4.5; >=4.5 is release-worthy and >=6 is a strong result.
Approximate WEPS must never rescue a failed exact/full-expert gate.

## Transfer and defer

Transfer now: rank-local loading, all-expert 512-wide TP3, KDA/MLA head sharding,
head-banded output/argmax, fused routed+shared MoE reduction, rank-parallel enqueue,
CUDA graphs, and SM121 retuning of IQ1_S kernels.

Defer: ragged 299/299/298 expert parallelism, peer-one-shot, multimem/NVLS,
FlashKDA for decode, DSpark before a fast target exists, vLLM GGUF, and bandwidth/MTU
tuning as the primary fix. The 8x H200 headline cannot be scaled to Sparks: aggregate
HBM bandwidth is roughly 38.4 TB/s there versus 0.819 TB/s here, before NVSwitch and
runtime differences.
