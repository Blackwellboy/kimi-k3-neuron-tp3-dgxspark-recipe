#!/usr/bin/env python3
import datetime, os, sys, time
import torch
import torch.distributed as dist

def main() -> int:
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    n = torch.cuda.device_count()
    if n < world:
        print(f"need {world} GPUs, saw {n}", flush=True)
        return 2
    torch.cuda.set_device(rank)
    addr = os.environ["MASTER_ADDR"]
    port = os.environ["MASTER_PORT"]
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{addr}:{port}",
        rank=rank,
        world_size=world,
        timeout=datetime.timedelta(seconds=120),
        device_id=torch.device(f"cuda:{rank}"),
    )
    expect = float(world * (world + 1) / 2)
    for count in (7168, 10752):
        t = torch.full((count,), float(rank + 1), device=f"cuda:{rank}", dtype=torch.float32)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        ok = bool(torch.allclose(t, torch.full_like(t, expect)))
        b = torch.full((count,), float(rank + 1), device=f"cuda:{rank}", dtype=torch.float32)
        for _ in range(20):
            dist.all_reduce(b, op=dist.ReduceOp.SUM)
            b.fill_(float(rank + 1))
        torch.cuda.synchronize(rank)
        iters = 300
        t0 = time.perf_counter()
        for _ in range(iters):
            dist.all_reduce(b, op=dist.ReduceOp.SUM)
            b.fill_(float(rank + 1))
        torch.cuda.synchronize(rank)
        us = (time.perf_counter() - t0) * 1e6 / iters
        print(f"rank{rank} count={count} bytes={count*4} correct={ok} mean_us={us:.3f} ms185={(us*185)/1000:.3f}", flush=True)
        if not ok:
            dist.destroy_process_group()
            return 1
    dist.destroy_process_group()
    print(f"rank{rank} PASS", flush=True)
    return 0

if __name__ == "__main__":
    sys.exit(main())
