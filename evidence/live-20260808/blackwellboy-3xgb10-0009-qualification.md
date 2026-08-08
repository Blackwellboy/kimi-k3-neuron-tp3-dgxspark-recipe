# Evidence: 3× NVIDIA DGX Spark / GB10 qualification of Kimi-K3 TP3 patches 0001–0009

**AUTHOR CODE:** Victor / vcruz305 (`kimi-k3-neuron-tp3-dgxspark-recipe`)  
**EXTERNAL HARDWARE QUALIFICATION:** Blackwellboy 3× DGX Spark fleet  

## Pins
| Item | Value |
|------|--------|
| Author commit | `e87c1c12829a079d2a6b45e5a840bd70f67e0f33` |
| SparkInfer base | `7a9b77a043596157d74e4af376cf9f29f68ce368` |
| Patch chain | **0001–0009** (exact `git am`, no local invent) |
| New patch 0009 | F16 `token_embd` / dist `embed_token` support |

## Hardware / topology
| Rank | Role | GPU |
|------|------|-----|
| 0 | coordinator + full LM head | NVIDIA GB10 |
| 1 | worker FFN [512,1024) | NVIDIA GB10 |
| 2 | worker FFN [1024,1536) | NVIDIA GB10 |

NCCL: RoCE / NET/IB, **auto GID** (do not force index 3; host GID tables differ).

## Prior 0008 boundary (same fleet)
- LoadReady fix **PASS**; 3/3 real load **PASS**
- First forward reached then deterministic failure: **`forward: embed_token failed`** (F16 token_embd)

## 0009 CPU / kernels
- loader: **38/38**
- dist generate protocol: **17/17**
- row dequant (F16): **PASS**
- proj GEMV (F16): **PASS**

## 0009 live 3-rank result
| Gate | Result |
|------|--------|
| Real load 3/3 | **PASS** (~32–38 min/rank; 896 experts; 512 FFN bands) |
| Duplicate LoadReady | **NO** |
| 0008 embed_token failure reproduced | **NO** |
| 0009 F16 fix behavior | **IMPROVED** (prior fail string absent) |
| First forward completion | **NO** |
| Generated tokens | **0** |
| Furthest **confirmed** boundary | **POST_LOAD / PRE-OR-DURING FIRST FORWARD** |
| Final status | **POST_LOAD_FIRST_FORWARD_STALL** |

Memory: min MemAvailable ~8.2–10.1 GiB; peak SwapUsed ~0.6–0.9 GiB; **no active swap thrash** at stall capture (si/so≈0).

Stall (~1h39m wall): all ranks sleeping (`wait_woken` / `hrtimer_nanosleep`) with NCCL IB helper threads and ESTAB TCP control + peer sockets; **no application log after `loaded ok`**. Exact wait object (**wait_token vs NCCL collective vs compute**) **UNKNOWN** without further author instrumentation.

## Claim boundaries
- Do **not** claim embed_token completed.
- Do **not** claim multi-Spark tok/s or production readiness.
- No author implementation changes in this evidence packet.

## Attribution trail
1. External 0008 qualification discovered post-LoadReady `embed_token failed`.  
2. 0009 is **Victor’s** author fix for F16 token embedding.  
3. External 0009 qualification shows load 3/3 + no prior embed error string, then **stall before proven forward/generate**.
