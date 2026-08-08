// Multi-process NCCL InitRank microbench for 3-Spark-style ranks.
//
// One process per rank. On a single 3-GPU host (pod stand-in for 3 Sparks):
//   CUDA_VISIBLE_DEVICES=0 ./k3_dist_nccl_microbench --rank 0 --world 3 --port 29500 &
//   CUDA_VISIBLE_DEVICES=1 ./k3_dist_nccl_microbench --rank 1 --world 3 --port 29500 &
//   CUDA_VISIBLE_DEVICES=2 ./k3_dist_nccl_microbench --rank 2 --world 3 --port 29500
//
// Rank 0 creates ncclUniqueId and serves it over TCP; others connect and join
// ncclCommInitRank. Measures f32 all-reduce latency at K3 payload sizes
// (7168 and 10752 floats = 28,672 / 43,008 bytes).
//
// Build (pod):
//   g++ -O2 -std=c++17 k3_dist_nccl_microbench.cpp -o k3_dist_nccl_microbench \
//       -I$CUDA_HOME/include -L$CUDA_HOME/lib64 -lcudart -lnccl -pthread
//
// Exit 0 only if correctness passes on every rank.

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cuda_runtime.h>
#include <nccl.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

void die(const char* msg) {
    std::fprintf(stderr, "FATAL: %s\n", msg);
    std::exit(2);
}

void cuda_check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA %s: %s\n", what, cudaGetErrorString(e));
        std::exit(2);
    }
}

void nccl_check(ncclResult_t e, const char* what) {
    if (e != ncclSuccess) {
        std::fprintf(stderr, "NCCL %s: %s\n", what, ncclGetErrorString(e));
        std::exit(2);
    }
}

bool send_all(int fd, const void* p, size_t n) {
    const char* c = static_cast<const char*>(p);
    size_t off = 0;
    while (off < n) {
        ssize_t w = ::send(fd, c + off, n - off, 0);
        if (w <= 0) return false;
        off += static_cast<size_t>(w);
    }
    return true;
}

bool recv_all(int fd, void* p, size_t n) {
    char* c = static_cast<char*>(p);
    size_t off = 0;
    while (off < n) {
        ssize_t r = ::recv(fd, c + off, n - off, 0);
        if (r <= 0) return false;
        off += static_cast<size_t>(r);
    }
    return true;
}

// Rank 0 listens; each worker receives the unique id bytes.
void exchange_id(int rank, int world, const char* host, int port, ncclUniqueId* id) {
    if (rank == 0) {
        nccl_check(ncclGetUniqueId(id), "ncclGetUniqueId");
        int ls = ::socket(AF_INET, SOCK_STREAM, 0);
        if (ls < 0) die("socket");
        int one = 1;
        setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(static_cast<uint16_t>(port));
        addr.sin_addr.s_addr = htonl(INADDR_ANY);
        if (bind(ls, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) die("bind");
        if (listen(ls, world) != 0) die("listen");
        std::printf("[rank0] listening on %d, serving nccl id to %d peers\n", port, world - 1);
        for (int i = 0; i < world - 1; ++i) {
            int c = accept(ls, nullptr, nullptr);
            if (c < 0) die("accept");
            if (!send_all(c, id, sizeof(*id))) die("send id");
            close(c);
        }
        close(ls);
    } else {
        // Retry connect briefly while rank0 binds.
        int fd = -1;
        for (int attempt = 0; attempt < 100; ++attempt) {
            fd = ::socket(AF_INET, SOCK_STREAM, 0);
            sockaddr_in addr{};
            addr.sin_family = AF_INET;
            addr.sin_port = htons(static_cast<uint16_t>(port));
            if (inet_pton(AF_INET, host, &addr.sin_addr) != 1) die("inet_pton");
            if (connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0) break;
            close(fd);
            fd = -1;
            usleep(50 * 1000);
        }
        if (fd < 0) die("connect to rank0");
        if (!recv_all(fd, id, sizeof(*id))) die("recv id");
        close(fd);
    }
}

double bench_allreduce(ncclComm_t comm, float* buf, size_t count, int iters, cudaStream_t stream) {
    // warmup
    for (int i = 0; i < 10; ++i) {
        nccl_check(ncclAllReduce(buf, buf, count, ncclFloat32, ncclSum, comm, stream), "warmup");
    }
    cuda_check(cudaStreamSynchronize(stream), "sync warmup");
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
        nccl_check(ncclAllReduce(buf, buf, count, ncclFloat32, ncclSum, comm, stream), "bench");
    }
    cuda_check(cudaStreamSynchronize(stream), "sync bench");
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
}

}  // namespace

int main(int argc, char** argv) {
    int rank = -1, world = 3, port = 29500, iters = 200;
    const char* host = "127.0.0.1";
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", name);
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--rank") rank = std::atoi(need("--rank"));
        else if (a == "--world") world = std::atoi(need("--world"));
        else if (a == "--port") port = std::atoi(need("--port"));
        else if (a == "--host") host = need("--host");
        else if (a == "--iters") iters = std::atoi(need("--iters"));
        else if (a == "--help") {
            std::printf("usage: %s --rank R --world 3 [--host 127.0.0.1] [--port 29500] [--iters 200]\n",
                        argv[0]);
            return 0;
        }
    }
    if (rank < 0 || rank >= world || world < 2) die("need --rank and --world");

    // Force NVLS off on some pods (same as prior K3 work).
    setenv("NCCL_NVLS_ENABLE", "0", 0);

    int ndev = 0;
    cuda_check(cudaGetDeviceCount(&ndev), "device count");
    if (ndev < 1) die("no CUDA device in this process (set CUDA_VISIBLE_DEVICES)");
    cuda_check(cudaSetDevice(0), "set device 0 (visible)");

    ncclUniqueId id{};
    exchange_id(rank, world, host, port, &id);

    ncclComm_t comm = nullptr;
    nccl_check(ncclCommInitRank(&comm, world, id, rank), "ncclCommInitRank");
    std::printf("[rank %d/%d] ncclCommInitRank OK\n", rank, world);

    cudaStream_t stream;
    cuda_check(cudaStreamCreate(&stream), "stream");

    // Correctness: each rank writes (rank+1); sum = world*(world+1)/2
    const size_t counts[] = {7168ull, 10752ull};  // 28,672 and 43,008 bytes
    const float expect = static_cast<float>(world * (world + 1) / 2);
    int fails = 0;

    for (size_t count : counts) {
        float* d = nullptr;
        cuda_check(cudaMalloc(&d, count * sizeof(float)), "malloc");
        std::vector<float> h(count, static_cast<float>(rank + 1));
        cuda_check(cudaMemcpy(d, h.data(), count * sizeof(float), cudaMemcpyHostToDevice), "H2D");
        nccl_check(ncclAllReduce(d, d, count, ncclFloat32, ncclSum, comm, stream), "correctness AR");
        cuda_check(cudaStreamSynchronize(stream), "sync corr");
        cuda_check(cudaMemcpy(h.data(), d, count * sizeof(float), cudaMemcpyDeviceToHost), "D2H");
        bool ok = true;
        for (size_t i = 0; i < count; ++i) {
            if (h[i] != expect) {
                ok = false;
                break;
            }
        }
        const double us = bench_allreduce(comm, d, count, iters, stream);
        std::printf("[rank %d] count=%zu bytes=%zu correct=%s p50_est_us=%.3f  (expect_sum=%.0f)\n",
                    rank, count, count * 4, ok ? "YES" : "NO", us, expect);
        if (!ok) ++fails;
        // Estimate ms/token at 185 reductions of the larger payload
        if (count == 10752ull && rank == 0) {
            std::printf("[rank0] rough 185x reductions => %.2f ms/token collective-only @ this size\n",
                        us * 185.0 / 1000.0);
        }
        cudaFree(d);
    }

    ncclCommDestroy(comm);
    cudaStreamDestroy(stream);
    if (fails) {
        std::printf("[rank %d] FAIL\n", rank);
        return 1;
    }
    std::printf("[rank %d] PASS\n", rank);
    return 0;
}
