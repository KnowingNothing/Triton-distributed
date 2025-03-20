# Triton-distributed

[Original Triton README](upstream-README.md) | [README in Chinese](README-cn.md)

Triton-distributed is a distributed compiler designed for computation-communication overlapping, which is based on OpenAI Triton.

Using Triton-distributed, programmers are able to develop efficient kernels comparable to highly-optimized libraries (including [Distributed-GEMM](https://github.com/NVIDIA/cutlass/tree/main/examples/65_distributed_gemm) and [FLUX](https://github.com/bytedance/flux/blob/main/README.md)).
Triton-distributed currently mainly targets Nvidia GPU and AMD GPU. It can also be ported to other hardware platforms.
Feel free to contact us if you want to use Triton-distributed on your own hardware.

## Getting started
### Install Triton-distributed from source
#### The best practice to use Triton-distributed:
- Python 3.9 (suggest using virtual environment)
- CUDA 12.4
- Torch 2.4

Dependencies with other versions may also work well, but this is not guaranteed. If you find any problem in installing, please tell us in Issues.

#### Steps:
1. Clone Triton-distributed to your own path (e.g., `/home/Triton-distributed`)
2. Update submodules
    ```sh
    git submodule update --init --recursive
    ```
3. Install dependencies
    ```sh
    pip3 install torch==2.4
    pip3 install black "clang-format==19.1.2" pre-commit ruff yapf==0.43
    pip3 install ninja cmake wheel pybind11 cuda-python==12.4 numpy chardet pytest
    ```
4. Build
    ```sh
    cd /home/Triton-distributed
    export USE_TRITON_DISTRIBUTED_AOT=0
    pip3 install -e python --verbose --no-build-isolation
    ```
    
    If you want to use AOT, then
    ```sh
    export USE_TRITON_DISTRIBUTED_AOT=1
    pip3 install -e python --verbose --no-build-isolation
    ```
    (Note: You have to first build non-AOT version before building AOT version)
5. Setup environment variables (Do this step at the beginning every time you use Triton-distributed)
    ```sh
    cd /home/Triton-distributed
    source scripts/setenv.sh
    ```

### Test your installation
#### AllGather GEMM example on single node
This example runs on a single node with 8 H800 GPUs.
```sh
bash ./third_party/distributed/launch.sh ./third_party/distributed/distributed/test/test_ag_gemm_intra_node.py --case correctness_tma
```
#### GEMM ReduceScatter example on single node
This example runs on a single node with 8 H800 GPUs.
```sh
bash ./third_party/distributed/launch.sh ./third_party/distributed/distributed/test/test_gemm_rs_intra_node.py 8192 8192 29568 --check
```
#### NVSHMEM example in Triton-distributed
```sh
bash ./third_party/distributed/launch.sh ./third_party/distributed/distributed/test/test_nvshmem_api.py
```

### How to use Triton-distributed
Triton-distributed provides a set of easy-to use primitives to support the development of distributed compute-communication overlapping kernels. The primitives are divided into low-level primitives and high-level primitives. Currently, we have released our low-level primitives, and we plan to release high-level primitives in future.

All the primitives are exposed by `triton.distributed.language`
#### Low-level primitives
##### Context Querying Primitives
```py
rank(axis=-1, _builder=None)
num_ranks(axis=-1, _builder=None)
symm_at(ptr, rank, _builder=None)

```
##### Singal Control Primitives
```py
wait(barrierPtrs, numBarriers, scope: str, semantic: str, _builder=None)
consume_token(value, token, _builder=None)
notify(ptr, rank, signal=1, sig_op="set", comm_scope="inter_node", _builder=None)
```
##### NVSHMEM-related Primitives

Besides the primitives, Triton-distributed also expose all the NVSHMEM primitives to Python, allowing users to program communication kernels purely in Python.

All the NVSHMEM-related device-side primitives are exposed by `triton.language.extra.libshmem_device`
```py
my_pe()
n_pes()
int_p(dest, value, pe)
remote_ptr(local_ptr, pe)
barrier_all()
barrier_all_block()
barrier_all_warp()
sync_all()
sync_all_block()
sync_all_warp()
quiet()
fence()
getmem_nbi_block(dest, source, bytes, pe)
getmem_block(dest, source, bytes, pe)
getmem_nbi_warp(dest, source, bytes, pe)
getmem_warp(dest, source, bytes, pe)
getmem_nbi(dest, source, bytes, pe)
getmem(dest, source, bytes, pe)
putmem_block(dest, source, bytes, pe)
putmem_nbi_block(dest, source, bytes, pe)
putmem_warp(dest, source, bytes, pe)
putmem_nbi_warp(dest, source, bytes, pe)
putmem(dest, source, bytes, pe)
putmem_nbi(dest, source, bytes, pe)
putmem_signal_nbi(dest, source, bytes, sig_addr, signal, sig_op, pe)
putmem_signal(dest, source, bytes, sig_addr, signal, sig_op, pe)
putmem_signal_nbi_block(dest, source, bytes, sig_addr, signal, sig_op, pe)
putmem_signal_block(dest, source, bytes, sig_addr, signal, sig_op, pe)
putmem_signal_nbi_warp(dest, source, bytes, sig_addr, signal, sig_op, pe)
putmem_signal_warp(dest, source, bytes, sig_addr, signal, sig_op, pe)
signal_op(sig_addr, signal, sig_op, pe)
signal_wait_until(sig_addr, cmp_, cmp_val)
```

Using these primitives, users can program compute-communication kernels easily. For example, a ring-put example is shown here:
```py
@triton.jit
def ring_put(ptr):
    mype = libshmem_device.my_pe()
    npes = libshmem_device.n_pes()
    peer = (mype + 1) % npes
    libshmem_device.int_p(ptr, mype, peer)
```

#### High-level primitives
To provide better programming experience, we also provide a set of high-level primitives for communication and signal control. These primitives, as decribed in our [MLSys 2025 paper](https://mlsys.org/virtual/2025/poster/2969), use a tile-centric design philosophy. These high-level primitives will be released soon after MLSys 2025.

## Roadmaps
### Functionalities
- [x] Release low-level primitives
- [ ] Release high-level primitives
### Kernels
- [x] Release single-node GEMM TP overlapping kernels
- [ ] Release single-node MoE TP overlapping kernels
- [ ] Release single-node distributed Flash-Decoding kernels
- [ ] Release single-node MoE EP overlapping kernels
- [ ] Release cross-node GEMM TP overlapping kernels
- [ ] Release cross-node MoE TP overlapping kernels
- [ ] Release cross-node distributed Flash-Decoding kernels
- [ ] Release cross-node EP all-to-all kernels (similar to [DeepEP](https://github.com/deepseek-ai/DeepEP))
### Backends
- [x] Nvidia SM90a support
- [x] Nvidia SM80 support
- [ ] Nvidia SM89 support
- [ ] AMD support
### Performance
- [ ] Performance report


## Citations
If you use Triton-distributed in a scientific publication, we encourage you to add the following reference to the related papers:
```bibtex
@misc{zheng2025tilelink,
      title={TileLink: Generating Efficient Compute-Communication Overlapping Kernels using Tile-Centric Primitives},
      author={Size Zheng, Jin Fang, Xuegui Zheng, Qi Hou, Wenlei Bao, Ningxin Zheng, Ziheng Jiang, Dongyang Wang, Jianxi Ye, Haibin Lin, Li-Wen Chang, Xin Liu},
      year={2025},
}
```

## License
The Triton-distributed project is under MIT license.