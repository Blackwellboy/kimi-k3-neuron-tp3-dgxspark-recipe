# 3-rank NCCL InitRank microbench receipt (2026-08-07)

## Budget

| Item | Value |
|---|---|
| Starting balance | ~$356.56 |
| Ending balance | ~$355.01 |
| Approx session spend | **~$1.55** |
| Bad 3×H200 attempt | $8.25/hr, ~8–10 min, GPUs 0/2 unavailable (ECC), deleted |
| Good 8×5090 run | $4.80/hr, ~10–15 min, deleted after PASS |
| Hard caps set | H200 2.5h; 5090 1.5h (both released early) |

## Result (production-relevant)

**3 independent processes, `nccl` via `torch.distributed` / InitRank-style TCP store, world=3, devices 0/1/2 on one host.**

| Payload | Bytes | Correct sum | mean latency | ×185 reductions |
|---|---:|---|---:|---:|
| 7168 f32 | 28,672 | YES | **~41.5 µs** | **~7.67 ms/token** |
| 10752 f32 | 43,008 | YES | **~41.5 µs** | **~7.69 ms/token** |

All three ranks: `PASS`.  
NCCL 2.29.7 + CUDA 13.2 (torch 2.12.0+cu130).  
`NCCL_NVLS_ENABLE=0`. IB wrapper warn only (no RDMA on this consumer box — expected).

Logs: `torch-nccl3-ranks.log`

## What this proves for 3-Spark bring-up

1. **One process per rank + NCCL InitRank-class join works** (not `ncclCommInitAll`).
2. At K3 collective sizes, **collective-only cost ~7.7 ms/token** on this PCIe 5090 node — same order as the earlier H200 same-host microprobe (~5.8–9 ms @ 185 calls).
3. This is **not** a full K3 decode number and **not** RoCE/Spark fabric evidence.

## What failed

- First rental `cd6fe737…` 3×H200 @ $8.25/hr: only GPU1 usable; GPUs 0 and 2 returned `cudaErrorDevicesUnavailable` with elevated uncorrectable ECC on GPU2. Pod deleted immediately.
- No other H200/H100 ≤$12/hr was listed after that. Full TP3 weight load (~113 GiB/rank) still needs a healthy multi-H200 (or 3 Sparks).

## Code frozen with this session

- `0003` transport/TP3–TP4 plans (prior)
- `0004-tp-rank-local-plan-and-nccl-microbench.patch` — rank-local plan + standalone C++ microbench source  
  SHA-256 `8fa18d0c9edc690e383b58ff9b088bec8e4c5037413db8150dec907d4897bace`
- CPU: `tp_rank_local_plan_cpu_test` 34/34

## Next (when healthy ≥3×H200 ≤$12/hr returns)

1. Re-run this NCCL microbench on H200 (expect similar or better µs).
2. Apply SparkInfer patch chain + `SPARKINFER_TP=ON` build.
3. Rank-local load smoke at ctx 8K, `WEPS=0`, no paging.
4. One-token parity vs llama, then multi-token.

Do **not** keep idle pods waiting for model download without a time box.
