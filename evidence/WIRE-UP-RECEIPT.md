# SparkInfer TP3 all-expert-width wire-up — 2026-08-07

## Conceptual map (why two runtimes)

| | **llama.cpp** (Unsloth K3 fork) | **SparkInfer-K3** |
|---|---|---|
| What it is | GGUF production runtime used for release serving, PPL, HumanEval, SparkBench | Separate high-performance **K3-native** runtime (not a llama.cpp fork in git ancestry) |
| Role for this project | Correctness / production path | Kernel + TP graph reference; same-host speed path for the released homogeneous IQ1_S GGUF |
| Three Sparks today | RPC **layer split** (~2.85 tok/s reported) | **Not** drop-in networked TP. Uses one process + `ncclCommInitAll` over **local** GPUs |
| matched-byte-v2 | Supported | Fail-closed rejected (Q6_K + mixed IQ2_XXS) |

SparkInfer pins Unsloth llama.cpp as a **reference engine for dumps/compare**, but its code base is Gittensor SparkInfer, MIT + NOTICE.

## What was implemented this session

Tree: `/root/k3/sparkinfer-tp3-width-run` @ upstream `7a9b77a` + patches:

1. fail-closed quant contract  
2. TP greedy generator CLI  
3. quant preflight follow-up  
4. F16 boundary + F16 preflight  
5. TP3 all-expert FFN-width **planner** (`0001-tp-plan-...`)  
6. **New wire-up** (executor):  
   - allow TP3 when `moe_ffn % 3 == 0` and block-aligned (512)  
   - call `moe_all_expert_width_dims` in `kimi_k3_tp_init`  
   - `k3_moe_ffn_local` honors `MoeShardMode::AllExpertsFfnWidth`  
   - CLI preflight accepts width path  

Local scripts: `release_eval/llama_fork/sparkinfer/tp3_width/wire_tp3_all_expert_width.py`, `patch_tp_generate_preflight.py`.

## Evidence

### CPU / unit
`ctest` 5/5: proj_gemv, row_dequant, tp_generate_cli, quant_contract, tp3_all_expert_width.

### Same-host TP3 load (3× H200)
With `NCCL_NVLS_ENABLE=0` (required on this pod; same lesson as earlier NCCL microprobe):

```
TP geometry: AllExpertsFfnWidth preflight OK (experts=896, moe_ffn=1536 -> 512/rank)
[k3-tp] rank 0: device 0, all 896 experts, ffn [0,512) (AllExpertsFfnWidth)
[k3-tp] rank 1: device 1, all 896 experts, ffn [512,1024) (AllExpertsFfnWidth)
[k3-tp] rank 2: device 2, all 896 experts, ffn [1024,1536) (AllExpertsFfnWidth)
[tp] all-reduce backend: nccl, 3 rank(s)
```

Weights load ~108 GiB/rank then NCCL Init COMPLETE.

### TP3 prefill — still blocked
```
[k3-prefill] tile at token 0 failed; the KV cache is already part-written, so there is no fallback
TP prompt prefill failed
```
Reproduced at full depth and `--max-layers 1`. Not a geometry reject anymore; next work is prefill/collective buffer ownership under AllExpertsFfnWidth.

### Layer-split multi-token (works today)
`kimi_k3_generate` on devices 0,1,2, prompt "The capital of France is", 8 new tokens, WEPS=0:

```
generated text:  Paris, which is a beautiful city with
decode: 7 forward passes in 0.289 s = 24.23 tok/s
```

First token matches prior one-token parity (`17374` / ` Paris`).

## Pod notes
- Stopped idle `llama-server` to free GPUs for SparkInfer; pod lifecycle preserved.
- Removal still owner-set (see CURRENT.md).
- Do not claim TP3 tok/s yet.

## Next SparkInfer steps (ordered)
1. Debug TP3 prefill tile failure under AllExpertsFfnWidth (owned collective buffers / expert-band vs width-band dispatch).  
2. One-token TP3 logits vs layer-split / llama.cpp once prefill works.  
3. Short decode speed with WEPS=0.  
4. Only then distributed rank bootstrap for three Sparks.
