#!/usr/bin/env python3
"""GPU health gate + optional multi-node NCCL rank for H100 bring-up."""
import datetime
import json
import os
import socket
import subprocess
import sys
import time


def sh(cmd: str) -> str:
    return subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.STDOUT)


def health() -> dict:
    out = {
        "host": socket.gethostname(),
        "ok": True,
        "errors": [],
        "gpus": [],
        "torch": None,
        "nvcc": None,
    }
    try:
        out["nvidia_smi"] = sh(
            "nvidia-smi --query-gpu=index,name,memory.total,memory.used,"
            "ecc.errors.uncorrected.volatile.total,ecc.errors.uncorrected.aggregate.total,"
            "retired_pages.sbe,retired_pages.dbe,persistence_mode "
            "--format=csv,noheader"
        ).strip()
    except Exception as e:
        out["ok"] = False
        out["errors"].append(f"nvidia-smi: {e}")
        return out

    try:
        import torch

        n = torch.cuda.device_count()
        out["torch"] = {
            "version": torch.__version__,
            "cuda": torch.version.cuda,
            "device_count": n,
        }
        for i in range(n):
            try:
                torch.cuda.set_device(i)
                x = torch.randn(1024, 1024, device=f"cuda:{i}")
                y = (x @ x).sum().item()
                free, total = torch.cuda.mem_get_info(i)
                out["gpus"].append(
                    {
                        "index": i,
                        "name": torch.cuda.get_device_name(i),
                        "matmul_ok": True,
                        "matmul_sum": float(y),
                        "free_bytes": int(free),
                        "total_bytes": int(total),
                    }
                )
                del x
                torch.cuda.empty_cache()
            except Exception as e:
                out["ok"] = False
                out["errors"].append(f"gpu{i}: {type(e).__name__}: {e}")
                out["gpus"].append({"index": i, "matmul_ok": False, "error": str(e)})
        if n < 1:
            out["ok"] = False
            out["errors"].append("no CUDA devices")
    except Exception as e:
        out["ok"] = False
        out["errors"].append(f"torch: {e}")

    try:
        out["nvcc"] = sh("nvcc --version 2>/dev/null | tail -1").strip() or "missing"
    except Exception:
        out["nvcc"] = "missing"

    return out


def nccl_rank() -> int:
    import torch
    import torch.distributed as dist

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(0)
    addr = os.environ["MASTER_ADDR"]
    port = os.environ["MASTER_PORT"]
    print(f"rank{rank} connecting to {addr}:{port}", flush=True)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{addr}:{port}",
        rank=rank,
        world_size=world,
        timeout=datetime.timedelta(seconds=180),
    )
    expect = float(world * (world + 1) / 2)
    for count in (7168, 10752):
        t = torch.full((count,), float(rank + 1), device="cuda", dtype=torch.float32)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        ok = bool(torch.allclose(t, torch.full_like(t, expect)))
        b = torch.full((count,), float(rank + 1), device="cuda", dtype=torch.float32)
        for _ in range(20):
            dist.all_reduce(b, op=dist.ReduceOp.SUM)
            b.fill_(float(rank + 1))
        torch.cuda.synchronize()
        iters = 200
        t0 = time.perf_counter()
        for _ in range(iters):
            dist.all_reduce(b, op=dist.ReduceOp.SUM)
            b.fill_(float(rank + 1))
        torch.cuda.synchronize()
        us = (time.perf_counter() - t0) * 1e6 / iters
        print(
            f"rank{rank} count={count} bytes={count*4} correct={ok} "
            f"mean_us={us:.3f} ms185={(us*185)/1000:.3f}",
            flush=True,
        )
        if not ok:
            dist.destroy_process_group()
            return 1
    dist.destroy_process_group()
    print(f"rank{rank} PASS", flush=True)
    return 0


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "health"
    if mode == "health":
        h = health()
        print(json.dumps(h, indent=2))
        return 0 if h["ok"] else 2
    if mode == "nccl":
        return nccl_rank()
    print("usage: health|nccl")
    return 2


if __name__ == "__main__":
    sys.exit(main())
