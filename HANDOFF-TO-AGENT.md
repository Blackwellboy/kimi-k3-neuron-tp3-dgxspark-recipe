# How to hand this to a Spark-controlling LLM agent

## Minimal prompt (copy-paste)

```text
You control my NVIDIA DGX Spark fleet for development.

Clone and read this repository first:
  https://github.com/vcruz305/kimi-k3-neuron-tp3-dgxspark-recipe

Mandatory:
  1. Open SPARK-AGENT.md and follow it as your operating contract.
  2. Open THREE-SPARK-TP3-RECIPE.md for geometry and ladder.
  3. Do not claim multi-Spark production inference is ready.
  4. Run scripts/gpu_health_gate.py on every Spark before long work.
  5. Never put API tokens or SSH keys in chat or commits.

Current task:
  <PASTE TASK — e.g. "Run 3-rank NCCL microbench on sparks A/B/C and file a receipt">
```

## Recommended agent defaults

- Working tree: clone of this repo + separate SparkInfer checkout for `git am`.
- Model weights: local path to gated HF download (not in git).
- Secrets: environment only.
- Deliverable: receipt markdown under `evidence/live-YYYYMMDD/` with hostnames, commands, hashes.

## Pairing with sparkfleet

If you also use `vcruz305/sparkfleet` for lifecycle/updates:

1. Use **sparkfleet** for OS/firmware/driver fleet ops.  
2. Use **this repo** for K3 Neuron TP3 bring-up only.  
3. Do not mix update windows with full-model TP experiments on the same nodes.
